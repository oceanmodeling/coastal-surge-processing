"""
Integration checks that every pipeline write function declares `time` as
the leading dimension on disk (for cdo compatibility) and propagates the
source calendar end-to-end, plus the compute_daily_max.py/
compute_monthly_max.py year-boundary carry regression (see
nc_metadata.apply_year_boundary_carry() and test_nc_metadata.py for the
underlying day_start()/month_start() unit tests).
"""
import cftime
import netCDF4 as nc
import numpy as np
import pandas as pd

import compute_daily_max as cdm
import compute_monthly_max as cmm
import detide_surge
import extract_outputs_to_shoreline_pts as extract
import nc_metadata


def _fake_node(n_nodes=3):
    idx = np.arange(n_nodes)
    return dict(
        model_name='ADCIRC',
        node_index=idx,
        node_lon=np.linspace(-80, -79, n_nodes),
        node_lat=np.linspace(25, 26, n_nodes),
        node_depth=np.full(n_nodes, 10.0),
        point_lon=np.linspace(-80, -79, n_nodes),
        point_lat=np.linspace(25, 26, n_nodes),
        dist_km=np.zeros(n_nodes),
        source_csv='test.csv',
    )


def test_extraction_writes_time_leading_and_propagates_calendar(tmp_path):
    n_nodes = 3
    times = pd.date_range('1978-01-01 01:00', periods=24, freq='h')
    year_data = np.random.default_rng(0).normal(size=(n_nodes, len(times))).astype(np.float32)
    node = _fake_node(n_nodes)
    out_path = tmp_path / 'hourly.nc'

    extract.write_hourly_year(
        out_path, n_nodes, node['node_index'], node['node_lon'],
        node['node_lat'], node['node_depth'], node['point_lon'],
        node['point_lat'], node['dist_km'], 'test.csv',
        nc_metadata.load_metadata(), times, year_data, 'ADCIRC',
        'proleptic_gregorian')

    ds = nc.Dataset(out_path)
    assert ds.variables['twl'].dimensions == ('time', 'node')
    assert ds.variables['time'].calendar == 'proleptic_gregorian'
    np.testing.assert_allclose(ds.variables['twl'][:].T, year_data)
    ds.close()

    # Regression check: compute_daily_max.py/compute_monthly_max.py's own
    # read_hourly_year() must transpose back to (n_nodes, n_times) now that
    # the on-disk convention is (time, node) -- this caught a real bug where
    # the read side wasn't updated to match the write side.
    _, daily_data, _ = cdm.read_hourly_year(out_path, 'twl')
    assert daily_data.shape == (n_nodes, len(times))
    np.testing.assert_allclose(daily_data, year_data)

    _, monthly_data, _ = cmm.read_hourly_year(out_path, 'twl')
    assert monthly_data.shape == (n_nodes, len(times))
    np.testing.assert_allclose(monthly_data, year_data)

    _, detide_data, detide_cal = detide_surge.read_hourly_year(out_path, 'twl')
    assert detide_data.shape == (n_nodes, len(times))
    assert detide_cal == 'proleptic_gregorian'
    np.testing.assert_allclose(detide_data, year_data)


def _write_adcirc_year(path, year, data, calendar='standard'):
    """One per-year hourly file stamped the way real ADCIRC zeta output
    often is: one step after cold start (01:00) through exactly next year's
    Jan 1 00:00:00 inclusive -- the pattern whose trailing instant
    nc_metadata.apply_year_boundary_carry() must carry into the next file
    rather than count toward this file's Dec 31."""
    times = pd.date_range(f'{year}-01-01 01:00', f'{year + 1}-01-01 00:00',
                          freq='h')
    n_nodes = data.shape[0]
    node = _fake_node(n_nodes)
    md = nc_metadata.load_metadata(
        None, cli_overrides={'group_name': 'G', 'climate_forcing': 'F',
                             'scenario': 'S', 'location': 'GESLA'})
    extract.write_hourly_year(
        path, n_nodes, node['node_index'], node['node_lon'], node['node_lat'],
        node['node_depth'], node['point_lon'], node['point_lat'],
        node['dist_km'], 'test.csv', md, times, data, 'ADCIRC', calendar)
    return times


def test_daily_max_carries_year_boundary_spillover_into_next_year(tmp_path):
    """Regression for the real ADCIRC pattern (01:00 -> next year's Jan 1
    00:00 inclusive): the trailing instant of one file belongs to the next
    file's Jan 1, not this file's Dec 31, and must be carried across the
    file boundary -- see nc_metadata.apply_year_boundary_carry(). Mirrors
    compute_daily_max.py's own main() loop rather than re-deriving its logic.
    """
    calendar = 'standard'
    n_nodes = 2
    spike = 100.0
    hourly = tmp_path / 'hourly'
    hourly.mkdir()

    times_1979 = pd.date_range('1979-01-01 01:00', '1980-01-01 00:00', freq='h')
    data_1979 = np.zeros((n_nodes, len(times_1979)), dtype=np.float32)
    data_1979[:, -1] = spike  # the trailing 1980-01-01 00:00 instant
    _write_adcirc_year(hourly / 'twl_1hr_G_F_S_GESLA_197901-197912.nc',
                       1979, data_1979, calendar)

    times_1980 = pd.date_range('1980-01-01 01:00', '1981-01-01 00:00', freq='h')
    data_1980 = np.zeros((n_nodes, len(times_1980)), dtype=np.float32)
    _write_adcirc_year(hourly / 'twl_1hr_G_F_S_GESLA_198001-198012.nc',
                       1980, data_1980, calendar)

    var = nc_metadata.VARIABLES['WaterLevel']['name']
    year_files = nc_metadata.discover_hourly_year_files(hourly, var)
    assert [y for y, _ in year_files] == [1979, 1980]

    days = []
    carry = None
    for year, path in year_files:
        times, data, calendar = cdm.read_hourly_year(path, 'twl')
        times, data, carry = nc_metadata.apply_year_boundary_carry(
            times, data, calendar, carry)
        time_hours_all = nc_metadata.hours_since_epoch(times, calendar)
        dates = nc_metadata.day_start(times, calendar)
        for date in sorted(np.unique(dates)):
            mask = dates == date
            days.append(cdm.compute_day_max(date, data[:, mask],
                                            time_hours_all[mask]))
    if carry is not None:
        date = cftime.datetime(carry['time'].year, 1, 1, calendar=calendar)
        hours = np.array(
            [nc_metadata.hours_since_epoch_scalar(carry['time'], calendar)])
        days.append(cdm.compute_day_max(date, carry['data'][:, None], hours))

    # 365 real days in 1979, 366 in 1980 (leap), plus the orphaned single
    # instant (1980's own trailing spillover, with no 1981 file to receive
    # it) finalized as its own one-hour day.
    assert len(days) == 365 + 366 + 1
    nc_metadata.raise_on_duplicate_periods(
        [d['date'] for d in days], 'day', 'instant')

    def _find(y, m, d):
        return next(day for day in days
                    if (day['date'].year, day['date'].month, day['date'].day)
                    == (y, m, d))

    # The spike -- 1979's file's own last record -- must land on 1980-01-01,
    # not 1979-12-31.
    assert _find(1980, 1, 1)['max_val'][0] == np.float32(spike)
    assert _find(1979, 12, 31)['max_val'][0] == np.float32(0.0)

    out_path = tmp_path / 'daily_max.nc'
    cdm.write_daily_max(out_path, _fake_node(n_nodes),
                        nc_metadata.load_metadata(), 'WaterLevel', days,
                        calendar)

    ds = nc.Dataset(out_path)
    assert ds.dimensions['time'].size == len(days)
    assert ds.variables['twl'].dimensions == ('time', 'node')
    assert ds.variables['time'].calendar == 'standard'
    ds.close()


def test_monthly_max_carries_year_boundary_spillover_into_next_year(tmp_path):
    """Same real ADCIRC boundary pattern as the daily-max regression above,
    but at month granularity: the trailing instant belongs to next year's
    January, not this year's December -- see
    nc_metadata.apply_year_boundary_carry(). Mirrors compute_monthly_max.py's
    own main() loop (using its actual compute_month_info()/_advance_month()/
    finalize()) rather than re-deriving its logic."""
    calendar = 'standard'
    n_nodes = 2
    spike = 100.0
    hourly = tmp_path / 'hourly'
    hourly.mkdir()

    times_1979 = pd.date_range('1979-01-01 01:00', '1980-01-01 00:00', freq='h')
    data_1979 = np.zeros((n_nodes, len(times_1979)), dtype=np.float32)
    data_1979[:, -1] = spike  # the trailing 1980-01-01 00:00 instant
    _write_adcirc_year(hourly / 'twl_1hr_G_F_S_GESLA_197901-197912.nc',
                       1979, data_1979, calendar)

    times_1980 = pd.date_range('1980-01-01 01:00', '1981-01-01 00:00', freq='h')
    data_1980 = np.zeros((n_nodes, len(times_1980)), dtype=np.float32)
    _write_adcirc_year(hourly / 'twl_1hr_G_F_S_GESLA_198001-198012.nc',
                       1980, data_1980, calendar)

    var = nc_metadata.VARIABLES['WaterLevel']['name']
    year_files = nc_metadata.discover_hourly_year_files(hourly, var)

    finalized = []
    prev = None
    carry = None
    for year, path in year_files:
        times, data, calendar = cmm.read_hourly_year(path, 'twl')
        times, data, carry = nc_metadata.apply_year_boundary_carry(
            times, data, calendar, carry)
        time_hours_all = nc_metadata.hours_since_epoch(times, calendar)
        months_key = nc_metadata.month_start(times, calendar)
        for month_val in sorted(np.unique(months_key)):
            mask = months_key == month_val
            cur = cmm.compute_month_info(month_val.year, month_val.month,
                                         data[:, mask], time_hours_all[mask])
            prev, _ = cmm._advance_month(prev, cur, finalized)
    if carry is not None:
        cur = cmm.compute_month_info(
            carry['time'].year, 1, carry['data'][:, None],
            np.array([nc_metadata.hours_since_epoch_scalar(carry['time'],
                                                            calendar)]))
        prev, _ = cmm._advance_month(prev, cur, finalized)
    if prev is not None:
        finalized.append(cmm.finalize(prev))

    # 12 real months in 1979, 12 in 1980, plus the orphaned single instant
    # (1980's own trailing spillover, with no 1981 file to receive it)
    # finalized as its own one-hour January 1981.
    assert len(finalized) == 12 + 12 + 1
    nc_metadata.raise_on_duplicate_periods(
        [cftime.datetime(m['year'], m['month'], 1, calendar='standard')
         for m in finalized], 'month', 'instant')

    def _find(y, m):
        return next(mo for mo in finalized if (mo['year'], mo['month']) == (y, m))

    # The spike -- 1979's file's own last record -- must land on January
    # 1980, not December 1979.
    assert _find(1980, 1)['max_val'][0] == np.float32(spike)
    assert _find(1979, 12)['max_val'][0] == np.float32(0.0)

    out_path = tmp_path / 'monthly_max.nc'
    cmm.write_monthly_max(out_path, _fake_node(n_nodes),
                          nc_metadata.load_metadata(), 'WaterLevel',
                          finalized, 0, calendar)

    ds = nc.Dataset(out_path)
    assert ds.dimensions['time'].size == len(finalized)
    assert ds.variables['twl'].dimensions == ('time', 'node')
    assert ds.variables['time'].calendar == 'standard'
    ds.close()


def test_detide_rejects_non_standard_calendar(tmp_path):
    path = tmp_path / 'twl_360day.nc'
    ds = nc.Dataset(path, 'w')
    ds.createDimension('node', 2)
    ds.createDimension('time', 2)
    v = ds.createVariable('time', 'f8', ('time',))
    v.units = nc_metadata.TIME_UNITS
    v.calendar = '360_day'
    v[:] = [0.0, 1.0]
    ds.createVariable('twl', 'f4', ('time', 'node'))[:] = np.zeros((2, 2))
    ds.close()

    try:
        detide_surge.read_hourly_year(path, 'twl')
        assert False, 'expected a ValueError for a non-standard calendar'
    except ValueError as e:
        assert '360_day' in str(e)


def _write_instant_year(path, year, n_nodes=3):
    """One per-year hourly file stamped the way a reanalysis sampled on the
    hour is: YYYY-01-01 00:00 through YYYY-12-31 23:00 inclusive."""
    times = pd.date_range(f'{year}-01-01 00:00', f'{year}-12-31 23:00', freq='h')
    data = np.random.default_rng(year).normal(
        size=(n_nodes, len(times))).astype(np.float32)
    node = _fake_node(n_nodes)
    md = nc_metadata.load_metadata(
        None, cli_overrides={'group_name': 'G', 'climate_forcing': 'F',
                             'scenario': 'S', 'location': 'GESLA'})
    extract.write_hourly_year(
        path, n_nodes, node['node_index'], node['node_lon'], node['node_lat'],
        node['node_depth'], node['point_lon'], node['point_lat'],
        node['dist_km'], 'test.csv', md, times, data, 'ADCIRC', 'standard')
    return times


def test_instant_convention_avoids_duplicate_months_across_year_files(tmp_path):
    """Regression for the CORA case: two adjacent per-year files whose first
    timestep is exactly YYYY-01-01 00:00. Under 'end' the second file re-opens
    the previous December; under 'instant' it does not."""
    hourly = tmp_path / 'hourly'
    hourly.mkdir()
    for year in (1979, 1980):
        _write_instant_year(
            hourly / f'twl_1hr_G_F_S_GESLA_{year}01-{year}12.nc', year)

    var = nc_metadata.VARIABLES['WaterLevel']['name']
    year_files = nc_metadata.discover_hourly_year_files(hourly, var)
    assert [y for y, _ in year_files] == [1979, 1980]

    keys_end, keys_inst = [], []
    for _, path in year_files:
        ds = nc.Dataset(str(path), 'r')
        times = nc_metadata.read_times(ds, 'time')
        ds.close()
        keys_end.extend(np.unique(
            nc_metadata.month_start(times, 'standard', convention='end')))
        keys_inst.extend(np.unique(
            nc_metadata.month_start(times, 'standard', convention='instant')))

    ok_end, dups_end = nc_metadata.periods_are_unique(keys_end)
    ok_inst, dups_inst = nc_metadata.periods_are_unique(keys_inst)

    # 'end' re-opens Dec 1979 -- one duplicate per year boundary -- and also
    # invents a Dec 1978 from the very first timestep.
    assert not ok_end
    assert [(d.year, d.month) for d in dups_end] == [(1979, 12)]
    assert (keys_end[0].year, keys_end[0].month) == (1978, 12)

    # 'instant' gives exactly the 24 real months, in order, no duplicates.
    assert ok_inst and dups_inst == []
    assert len(keys_inst) == 24
    assert (keys_inst[0].year, keys_inst[0].month) == (1979, 1)
    assert (keys_inst[-1].year, keys_inst[-1].month) == (1980, 12)


def test_instant_convention_gives_exact_day_counts(tmp_path):
    """365 days for a non-leap year, 366 for a leap year, no spillover."""
    hourly = tmp_path / 'hourly'
    hourly.mkdir()
    for year in (1979, 1980):
        _write_instant_year(
            hourly / f'twl_1hr_G_F_S_GESLA_{year}01-{year}12.nc', year)
    var = nc_metadata.VARIABLES['WaterLevel']['name']
    counts = {}
    for year, path in nc_metadata.discover_hourly_year_files(hourly, var):
        ds = nc.Dataset(str(path), 'r')
        times = nc_metadata.read_times(ds, 'time')
        ds.close()
        counts[year] = len(np.unique(
            nc_metadata.day_start(times, 'standard', convention='instant')))
    assert counts == {1979: 365, 1980: 366}

"""
Integration checks that every pipeline write function declares `time` as
the leading dimension on disk (for cdo compatibility) and propagates the
source calendar end-to-end, plus the compute_daily_max.py 366-day
regression (see test_nc_metadata.py for the underlying day_start() unit
tests).
"""
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


def test_daily_max_gives_365_days_and_time_leading_dims(tmp_path):
    calendar = 'standard'
    n_nodes = 2
    times = pd.date_range('1978-01-01 01:00', '1979-01-01 00:00', freq='h')
    data = np.random.default_rng(1).normal(size=(n_nodes, len(times)))
    time_hours = nc_metadata.hours_since_epoch(times, calendar)
    dates = nc_metadata.day_start(times, calendar)

    days = [cdm.compute_day_max(date, data[:, dates == date],
                                time_hours[dates == date])
            for date in sorted(np.unique(dates))]
    assert len(days) == 365  # not 366 -- see the issue this fixes

    out_path = tmp_path / 'daily_max.nc'
    cdm.write_daily_max(out_path, _fake_node(n_nodes),
                        nc_metadata.load_metadata(), 'WaterLevel', days,
                        calendar)

    ds = nc.Dataset(out_path)
    assert ds.dimensions['time'].size == 365
    assert ds.variables['twl'].dimensions == ('time', 'node')
    assert ds.variables['time'].calendar == 'standard'
    ds.close()


def test_monthly_max_puts_spillover_hour_in_correct_december(tmp_path):
    calendar = 'standard'
    n_nodes = 2
    times = pd.date_range('1978-01-01 01:00', '1979-01-01 00:00', freq='h')
    data = np.random.default_rng(2).normal(size=(n_nodes, len(times)))
    time_hours = nc_metadata.hours_since_epoch(times, calendar)
    months_key = nc_metadata.month_start(times, calendar)

    unique_months = sorted(np.unique(months_key))
    assert len(unique_months) == 12  # 1978 only, not a spurious Jan 1979
    assert unique_months[-1] == __import__('cftime').datetime(
        1978, 12, 1, calendar=calendar)

    finalized = []
    prev = None
    for month_val in unique_months:
        mask = months_key == month_val
        cur = cmm.compute_month_info(month_val.year, month_val.month,
                                     data[:, mask], time_hours[mask])
        if prev is not None:
            cmm.resolve_adjacency(prev, cur)
            finalized.append(cmm.finalize(prev))
        prev = cur
    finalized.append(cmm.finalize(prev))

    out_path = tmp_path / 'monthly_max.nc'
    cmm.write_monthly_max(out_path, _fake_node(n_nodes),
                          nc_metadata.load_metadata(), 'WaterLevel',
                          finalized, 0, calendar)

    ds = nc.Dataset(out_path)
    assert ds.dimensions['time'].size == 12
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

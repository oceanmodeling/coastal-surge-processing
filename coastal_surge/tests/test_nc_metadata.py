from datetime import timedelta

import cftime
import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest

import nc_metadata


def test_day_start_shifts_exact_midnight_to_previous_day_under_end():
    # 'end' is no longer the default (see DEFAULT_TIME_STAMP_CONVENTION), so
    # this convention must now be requested explicitly.
    times = np.array([
        cftime.datetime(1978, 12, 31, 23, calendar='standard'),
        cftime.datetime(1979, 1, 1, 0, calendar='standard'),
    ], dtype=object)
    days = nc_metadata.day_start(times, 'standard', convention='end')
    assert days[0] == days[1] == cftime.datetime(1978, 12, 31, calendar='standard')


def test_day_start_is_a_noop_away_from_midnight():
    times = np.array([cftime.datetime(1978, 6, 15, 13, calendar='standard')],
                     dtype=object)
    days = nc_metadata.day_start(times, 'standard')
    assert days[0] == cftime.datetime(1978, 6, 15, calendar='standard')


def test_day_start_defaults_to_instant():
    """The default convention is 'instant': exactly midnight belongs to the
    day that begins, not the one that just ended."""
    times = np.array([cftime.datetime(1979, 1, 1, 0, calendar='standard')],
                     dtype=object)
    assert nc_metadata.day_start(times, 'standard')[0] == \
        cftime.datetime(1979, 1, 1, calendar='standard')


def test_full_nonleap_year_hour_shifted_end_convention_gives_365_days():
    # A solo array shaped like the real ADCIRC per-year pattern (first output
    # one step after cold start, last output landing exactly on next year's
    # Jan 1 00:00:00). Under 'end' this collapses to 365 days by construction
    # (the boundary instant is folded back into Dec 31); under the new
    # 'instant' default it's legitimately a distinct (partial) 366th day
    # belonging to the *next* file -- see
    # nc_metadata.apply_year_boundary_carry() and
    # test_dimension_order_and_calendar.py's cross-file regression, which
    # covers the multi-file case this solo array can't.
    times = pd.date_range('1978-01-01 01:00', '1979-01-01 00:00', freq='h')
    assert len(times) == 8760
    days_end = nc_metadata.day_start(times, 'standard', convention='end')
    assert len(np.unique(days_end)) == 365
    days_instant = nc_metadata.day_start(times, 'standard', convention='instant')
    assert len(np.unique(days_instant)) == 366


def test_month_start_360_day_boundary_goes_to_prior_month_under_end():
    times = np.array([
        cftime.datetime(1978, 2, 30, 23, calendar='360_day'),
        cftime.datetime(1978, 3, 1, 0, calendar='360_day'),
    ], dtype=object)
    months = nc_metadata.month_start(times, '360_day', convention='end')
    assert months[0] == months[1] == cftime.datetime(1978, 2, 1, calendar='360_day')


def test_hours_since_epoch_standard_calendar():
    times = pd.date_range('1900-01-01', periods=3, freq='h')
    hours = nc_metadata.hours_since_epoch(times, 'standard')
    np.testing.assert_allclose(hours, [0.0, 1.0, 2.0])


def test_read_times_360_day_does_not_crash_on_feb_30(tmp_path):
    """The bug this guards: pd.to_datetime can't represent 360_day's Feb 30
    (it raises DateParseError), which used to crash read_times()."""
    path = tmp_path / 'in.nc'
    ds = nc.Dataset(path, 'w')
    ds.createDimension('time', 3)
    v = ds.createVariable('time', 'f8', ('time',))
    v.units = nc_metadata.TIME_UNITS
    v.calendar = '360_day'
    dates = [cftime.datetime(1978, 2, 29, calendar='360_day'),
             cftime.datetime(1978, 2, 30, calendar='360_day'),
             cftime.datetime(1978, 3, 1, calendar='360_day')]
    v[:] = nc.date2num(dates, v.units, v.calendar)
    ds.close()

    ds = nc.Dataset(path, 'r')
    times = nc_metadata.read_times(ds, 'time')
    ds.close()

    assert isinstance(times, np.ndarray) and times.dtype == object
    assert (times[1].year, times[1].month, times[1].day) == (1978, 2, 30)


def test_write_times_360_day_round_trip(tmp_path):
    dates = np.array([
        cftime.datetime(1978, 2, 29, calendar='360_day'),
        cftime.datetime(1978, 2, 30, calendar='360_day'),
        cftime.datetime(1978, 3, 1, calendar='360_day'),
    ], dtype=object)

    out_path = tmp_path / 'out.nc'
    ds = nc.Dataset(out_path, 'w')
    ds.createDimension('time', 3)
    v = ds.createVariable('time', 'f8', ('time',))
    v.units = nc_metadata.TIME_UNITS
    v.calendar = '360_day'
    v[:] = nc_metadata.write_times(ds, 'time', dates)
    ds.close()

    ds = nc.Dataset(out_path, 'r')
    assert ds.variables['time'].calendar == '360_day'
    round_tripped = nc_metadata.read_times(ds, 'time')
    ds.close()
    for original, restored in zip(dates, round_tripped):
        assert original == restored


def test_read_times_standard_calendar_still_returns_datetimeindex(tmp_path):
    """No behavior change for the already-validated standard-calendar path."""
    path = tmp_path / 'std.nc'
    ds = nc.Dataset(path, 'w')
    ds.createDimension('time', 2)
    v = ds.createVariable('time', 'f8', ('time',))
    v.units = nc_metadata.TIME_UNITS
    v.calendar = 'standard'
    v[:] = [0.0, 1.0]
    ds.close()

    ds = nc.Dataset(path, 'r')
    times = nc_metadata.read_times(ds, 'time')
    ds.close()
    assert isinstance(times, pd.DatetimeIndex)


def _write_2d_var(path, dims, shape):
    ds = nc.Dataset(path, 'w')
    ds.createDimension('time', shape[dims.index('time')])
    ds.createDimension('node', shape[dims.index('node')])
    v = ds.createVariable('twl', 'f4', dims)
    v[:] = np.arange(np.prod(shape)).reshape(shape)
    ds.close()


def test_read_node_major_variable_transposes_time_node_order(tmp_path):
    path = tmp_path / 'time_node.nc'
    on_disk = np.arange(15).reshape(5, 3)  # (time, node)
    _write_2d_var(path, ('time', 'node'), (5, 3))
    ds = nc.Dataset(path, 'r')
    data = nc_metadata.read_node_major_variable(ds, 'twl')
    ds.close()
    assert data.shape == (3, 5)
    np.testing.assert_array_equal(data, on_disk.T)


def test_read_node_major_variable_leaves_node_time_order_as_is(tmp_path):
    """A file written by a pre-fix version of this pipeline, using the
    opposite on-disk convention, must not be silently transposed."""
    path = tmp_path / 'node_time.nc'
    expected = np.arange(15).reshape(3, 5)
    _write_2d_var(path, ('node', 'time'), (3, 5))
    ds = nc.Dataset(path, 'r')
    data = nc_metadata.read_node_major_variable(ds, 'twl')
    ds.close()
    assert data.shape == (3, 5)
    np.testing.assert_array_equal(data, expected)


def test_read_node_major_variable_rejects_unrecognized_dims(tmp_path):
    path = tmp_path / 'other.nc'
    ds = nc.Dataset(path, 'w')
    ds.createDimension('x', 3)
    ds.createDimension('y', 5)
    v = ds.createVariable('twl', 'f4', ('x', 'y'))
    v[:] = np.zeros((3, 5))
    ds.close()

    ds = nc.Dataset(path, 'r')
    try:
        nc_metadata.read_node_major_variable(ds, 'twl')
        assert False, 'expected a ValueError for unrecognized dimensions'
    except ValueError as e:
        assert 'twl' in str(e)
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# Time-stamping convention
# ---------------------------------------------------------------------------

def test_convention_epsilon_end_shifts_and_instant_does_not():
    assert nc_metadata.convention_epsilon('end') == timedelta(seconds=1)
    assert nc_metadata.convention_epsilon('instant') == timedelta(0)
    assert nc_metadata.convention_epsilon(None) == \
        nc_metadata.convention_epsilon(nc_metadata.DEFAULT_TIME_STAMP_CONVENTION)


def test_convention_epsilon_rejects_unknown():
    with pytest.raises(ValueError, match='Unknown time-stamping convention'):
        nc_metadata.convention_epsilon('middle')


def test_instant_convention_keeps_midnight_in_its_own_day():
    """The whole point of the flag: an instantaneous sample at midnight
    belongs to the day that begins, not the one that ended."""
    times = pd.to_datetime(['2000-03-01 00:00', '2000-03-01 01:00'])
    end = nc_metadata.day_start(times, 'standard', convention='end')
    inst = nc_metadata.day_start(times, 'standard', convention='instant')
    assert (end[0].year, end[0].month, end[0].day) == (2000, 2, 29)
    assert (inst[0].year, inst[0].month, inst[0].day) == (2000, 3, 1)
    assert (end[1].day, inst[1].day) == (1, 1)


def test_instant_convention_keeps_month_boundary_in_its_own_month():
    times = pd.to_datetime(['2001-01-01 00:00', '2001-01-01 05:00'])
    end = nc_metadata.month_start(times, 'standard', convention='end')
    inst = nc_metadata.month_start(times, 'standard', convention='instant')
    assert (end[0].year, end[0].month) == (2000, 12)
    assert (inst[0].year, inst[0].month) == (2001, 1)


def test_explicit_epsilon_still_overrides_convention():
    times = pd.to_datetime(['2000-03-01 00:00'])
    got = nc_metadata.day_start(times, 'standard', convention='end',
                                epsilon=timedelta(0))
    assert (got[0].month, got[0].day) == (3, 1)


def test_is_year_start_instant():
    assert nc_metadata.is_year_start_instant(
        cftime.datetime(1980, 1, 1, 0, calendar='standard'))
    assert not nc_metadata.is_year_start_instant(
        cftime.datetime(1980, 1, 1, 1, calendar='standard'))
    assert not nc_metadata.is_year_start_instant(
        cftime.datetime(1979, 12, 31, 0, calendar='standard'))


def test_apply_year_boundary_carry_is_a_noop_with_no_carry_and_no_spillover():
    times = pd.date_range('1980-01-01 00:00', '1980-12-31 23:00', freq='h')
    data = np.arange(2 * len(times), dtype=np.float64).reshape(2, len(times))
    out_times, out_data, new_carry = nc_metadata.apply_year_boundary_carry(
        times, data, 'standard', None)
    assert len(out_times) == len(times)
    np.testing.assert_array_equal(out_data, data)
    assert new_carry is None


def test_apply_year_boundary_carry_strips_trailing_year_start_instant():
    times = pd.date_range('1979-01-01 01:00', '1980-01-01 00:00', freq='h')
    data = np.zeros((2, len(times)))
    data[:, -1] = 42.0
    out_times, out_data, new_carry = nc_metadata.apply_year_boundary_carry(
        times, data, 'standard', None)
    assert len(out_times) == len(times) - 1
    assert out_times[-1] == pd.Timestamp('1979-12-31 23:00')
    assert new_carry is not None
    assert new_carry['time'] == pd.Timestamp('1980-01-01 00:00')
    np.testing.assert_array_equal(new_carry['data'], [42.0, 42.0])


def test_apply_year_boundary_carry_prepends_to_next_file():
    carry = {'time': pd.Timestamp('1980-01-01 00:00'),
            'data': np.array([42.0, 42.0])}
    times = pd.date_range('1980-01-01 01:00', '1980-01-01 03:00', freq='h')
    data = np.ones((2, len(times)))
    out_times, out_data, new_carry = nc_metadata.apply_year_boundary_carry(
        times, data, 'standard', carry)
    assert out_times[0] == pd.Timestamp('1980-01-01 00:00')
    np.testing.assert_array_equal(out_data[:, 0], [42.0, 42.0])
    np.testing.assert_array_equal(out_data[:, 1:], data)
    assert new_carry is None  # this file's own last record isn't a spillover


def test_apply_year_boundary_carry_duplicate_instant_is_just_prepended():
    """A year that (unlike the usual +1h pattern) starts its own file with
    00:00 -- duplicating what the previous file already spilled over -- is
    not deduplicated: it's harmless for a maximum, so the instant simply
    appears twice in this file's leading day/month group."""
    carry = {'time': pd.Timestamp('1980-01-01 00:00'),
            'data': np.array([42.0, 42.0])}
    times = pd.date_range('1980-01-01 00:00', '1980-01-01 02:00', freq='h')
    data = np.array([[7.0, 1.0, 1.0], [7.0, 1.0, 1.0]])
    out_times, out_data, new_carry = nc_metadata.apply_year_boundary_carry(
        times, data, 'standard', carry)
    assert len(out_times) == len(times) + 1
    assert list(out_times[:2]) == [pd.Timestamp('1980-01-01 00:00')] * 2
    np.testing.assert_array_equal(out_data[:, 0], [42.0, 42.0])
    np.testing.assert_array_equal(out_data[:, 1], [7.0, 7.0])


def test_periods_are_unique_detects_a_reopened_period():
    mk = lambda y, m, d: cftime.datetime(y, m, d, calendar='standard')
    ok, dups = nc_metadata.periods_are_unique(
        [mk(1979, 12, 1), mk(1980, 1, 1), mk(1979, 12, 1)])
    assert not ok
    assert [(d.year, d.month) for d in dups] == [(1979, 12)]
    ok, dups = nc_metadata.periods_are_unique([mk(1979, 12, 1), mk(1980, 1, 1)])
    assert ok and dups == []


def test_raise_on_duplicate_periods_names_the_other_convention():
    mk = lambda y, m, d: cftime.datetime(y, m, d, calendar='standard')
    with pytest.raises(ValueError, match='--time-stamp-convention instant'):
        nc_metadata.raise_on_duplicate_periods(
            [mk(1979, 12, 1), mk(1979, 12, 1)], 'month', 'end')
    nc_metadata.raise_on_duplicate_periods([mk(1979, 12, 1)], 'month', 'end')


def test_resolve_time_stamp_convention_defaults_and_validates():
    assert nc_metadata.resolve_time_stamp_convention({}) == \
        nc_metadata.DEFAULT_TIME_STAMP_CONVENTION
    assert nc_metadata.resolve_time_stamp_convention(
        {'time_stamp_convention': 'instant'}) == 'instant'
    with pytest.raises(ValueError):
        nc_metadata.resolve_time_stamp_convention(
            {'time_stamp_convention': 'nonsense'})


def test_convention_round_trips_through_metadata_and_file(tmp_path):
    """Set via CLI override -> written as a global attribute -> inherited."""
    md = nc_metadata.load_metadata(
        None, cli_overrides={'time_stamp_convention': 'instant',
                             'group_name': 'G', 'climate_forcing': 'F',
                             'scenario': 'S', 'location': 'GESLA'})
    assert md['time_stamp_convention'] == 'instant'
    path = tmp_path / 'x.nc'
    ds = nc.Dataset(str(path), 'w', format='NETCDF4')
    nc_metadata.set_global_attrs(ds, md, title='t', summary='s',
                                 timestep='MonthlyMax',
                                 variable_key='StormSurge')
    ds.close()
    assert nc_metadata.read_known_attrs(path)['time_stamp_convention'] == 'instant'


def test_roms_two_dimensional_node_index_round_trips(tmp_path):
    """MET Norway's ROMS submission indexes a curvilinear grid by a 0-based
    (i, j) pair; write_node_index/read_node_index must preserve it."""
    path = tmp_path / 'roms.nc'
    idx = np.array([[2, 0], [7, 11], [519, 791]])
    ds = nc.Dataset(str(path), 'w', format='NETCDF4')
    ds.createDimension('node', idx.shape[0])
    nc_metadata.write_node_index(ds, 'ROMS', idx)
    ds.close()

    ds = nc.Dataset(str(path), 'r')
    assert set(('node_i', 'node_j')).issubset(ds.variables)
    assert 'node_index' not in ds.variables
    assert ds.variables['node_i'].cf_role == 'timeseries_id'
    model_name, got = nc_metadata.read_node_index(ds)
    ds.close()
    assert model_name == 'ROMS'
    np.testing.assert_array_equal(got, idx)


def test_roms_scheme_is_zero_based_two_dimensional():
    assert nc_metadata.get_node_index_scheme('ROMS') == {'dims': 2, 'base': 0}

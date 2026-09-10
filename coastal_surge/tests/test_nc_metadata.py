import cftime
import netCDF4 as nc
import numpy as np
import pandas as pd

import nc_metadata


def test_day_start_shifts_exact_midnight_to_previous_day():
    times = np.array([
        cftime.datetime(1978, 12, 31, 23, calendar='standard'),
        cftime.datetime(1979, 1, 1, 0, calendar='standard'),
    ], dtype=object)
    days = nc_metadata.day_start(times, 'standard')
    assert days[0] == days[1] == cftime.datetime(1978, 12, 31, calendar='standard')


def test_day_start_is_a_noop_away_from_midnight():
    times = np.array([cftime.datetime(1978, 6, 15, 13, calendar='standard')],
                     dtype=object)
    days = nc_metadata.day_start(times, 'standard')
    assert days[0] == cftime.datetime(1978, 6, 15, calendar='standard')


def test_full_nonleap_year_hour_shifted_convention_gives_365_days():
    # ADCIRC per-year file convention: first output is one step after cold
    # start, last output lands exactly on next year's Jan 1 00:00:00 -- see
    # compute_daily_max.py giving 366 days for 1978 before this fix.
    times = pd.date_range('1978-01-01 01:00', '1979-01-01 00:00', freq='h')
    assert len(times) == 8760
    days = nc_metadata.day_start(times, 'standard')
    assert len(np.unique(days)) == 365


def test_month_start_360_day_boundary_goes_to_prior_month():
    times = np.array([
        cftime.datetime(1978, 2, 30, 23, calendar='360_day'),
        cftime.datetime(1978, 3, 1, 0, calendar='360_day'),
    ], dtype=object)
    months = nc_metadata.month_start(times, '360_day')
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

#!/usr/bin/env python3
"""
Compute full-period monthly maxima from a directory of per-year hourly
NetCDF files (either twl or ssgh), writing a single output file that spans
the whole record.

This script is Step 3 of the SurgeMIP water level extraction pipeline, and
can be run on the output of either Step 1 (twl) or Step 2 (ssgh):

  Step 1 — extract_outputs_to_shoreline_pts.py   -> hourly twl, per year
  Step 2 — detide_surge.py (optional)            -> hourly ssgh, per year
  Step 3 — this script                           -> full-period MonthlyMax

24-hour separation rule (per Natacha/Melissa):
  For each node, the timestamps of two chronologically adjacent months'
  maxima must be separated by at least 24 hours. In the rare case where
  they are not:
    - the larger of the two maxima is kept unchanged, and
    - for the month with the *smaller* maximum, that specific timestep is
      excluded and the month's maximum is recomputed from the remaining
      data.
  The recomputed value is then re-checked against its other neighbor (the
  check cascades, bounded to a few iterations — true multi-hop cascades are
  expected to be exceedingly rare). Every node/month whose maximum was
  altered by this rule is flagged 1 in the `<var>_adjusted` output variable,
  and the count is recorded in the `n_adjusted_node_months` global attribute,
  supporting a sensitivity discussion of this choice.

This is a single streaming pass (not checkpointed): each year's hourly file
is read once, in chronological order, keeping only the current and previous
month's raw hourly data in memory (enough to resolve the adjacency rule
across month and year boundaries) plus the finalized max/time/adjusted
arrays for every month completed so far. If interrupted, rerun from
scratch — this pass is far cheaper than the Step 2 tidal fit.

Usage:
  python compute_monthly_max.py \\
      --hourly-dir /path/to/twl_hourly/ \\
      --variable   WaterLevel \\
      --output-dir /path/to/monthly_max/

Dependencies:
  numpy, netCDF4, pandas
"""

import argparse
import sys
import time as timer
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd

import nc_metadata

MIN_SEPARATION_HOURS = 24.0
MAX_CASCADE_ITER = 5

# CF fill value written by extract_outputs_to_shoreline_pts.py / detide_surge.py
CF_FILL_F32 = 9.96921e+36


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        '--hourly-dir', required=True,
        help='Directory of per-year hourly NetCDF files (twl or ssgh) to '
             'compute monthly maxima from',
    )
    p.add_argument(
        '--variable', required=True, choices=sorted(nc_metadata.VARIABLES),
        help='Which variable to process: WaterLevel (twl) or StormSurge (ssgh)',
    )
    p.add_argument(
        '--output-dir', required=True,
        help='Directory to write the single full-period MonthlyMax NetCDF into',
    )
    p.add_argument(
        '--force', action='store_true',
        help='Overwrite the output file if it already exists',
    )
    p.add_argument(
        '--metadata-yaml',
        help='Path to a YAML file with global NetCDF metadata. Naming '
             'fields (group_name/climate_forcing/scenario/location) are '
             'otherwise inherited from the input hourly files. See '
             'metadata_template.yaml for the editable template.',
    )
    nc_metadata.add_naming_args(p)
    return p.parse_args()


def read_hourly_year(path, var_name):
    ds = nc.Dataset(str(path), 'r')
    times = nc_metadata.read_times(ds, 'time')
    data = np.array(ds.variables[var_name][:, :], dtype=np.float64)
    ds.close()
    return times, data


def read_node_metadata(path):
    ds = nc.Dataset(str(path), 'r')
    node_index = np.array(ds.variables['node_index'][:], dtype=np.int64) - 1
    node_lon   = np.array(ds.variables['node_lon'][:], dtype=np.float64)
    node_lat   = np.array(ds.variables['node_lat'][:], dtype=np.float64)
    node_depth = np.array(ds.variables['node_depth'][:], dtype=np.float64)
    point_lon  = np.array(ds.variables['point_lon'][:], dtype=np.float64)
    point_lat  = np.array(ds.variables['point_lat'][:], dtype=np.float64)
    dist_km    = np.array(ds.variables['dist_km'][:], dtype=np.float64)
    source_csv = getattr(ds, 'source_csv', '')
    ds.close()
    return dict(node_index=node_index, node_lon=node_lon, node_lat=node_lat,
                node_depth=node_depth, point_lon=point_lon,
                point_lat=point_lat, dist_km=dist_km, source_csv=source_csv)


# ---------------------------------------------------------------------------
# Per-month state
# ---------------------------------------------------------------------------

def compute_month_info(year, month, month_data, month_time_hours):
    """
    month_data : (n_nodes, n_hours) float64, CF_FILL_F32 for invalid/dry
    month_time_hours : (n_hours,) float64, hours since nc_metadata.EPOCH

    Returns a dict tracking enough state to both report this month's max
    and, if needed, exclude a timestep and recompute it later.
    """
    n_nodes = month_data.shape[0]
    valid = month_data < (CF_FILL_F32 / 2)
    data = np.where(valid, month_data, -np.inf)

    argmax_idx = np.argmax(data, axis=1)
    any_valid = valid.any(axis=1)
    max_val = data[np.arange(n_nodes), argmax_idx]
    max_val = np.where(any_valid, max_val, np.nan)
    max_time_hours = np.where(any_valid, month_time_hours[argmax_idx], np.nan)

    return {
        'year': year, 'month': month,
        'data': data,                    # (n_nodes, n_hours), mutated on recompute
        'time_hours': month_time_hours,  # (n_hours,)
        'argmax_idx': argmax_idx,        # (n_nodes,)
        'max_val': max_val,              # (n_nodes,)
        'max_time_hours': max_time_hours,
        'adjusted': np.zeros(n_nodes, dtype=bool),
    }


def _recompute_excluding_argmax(info, node_mask):
    """Exclude the current per-node argmax timestep and recompute the max
    from the remaining hours, for nodes where node_mask is True."""
    rows = np.where(node_mask)[0]
    info['data'][rows, info['argmax_idx'][rows]] = -np.inf

    new_argmax = np.argmax(info['data'][rows], axis=1)
    new_val = info['data'][rows, new_argmax]
    any_valid = np.isfinite(new_val)

    info['argmax_idx'][rows] = new_argmax
    info['max_val'][rows] = np.where(any_valid, new_val, np.nan)
    info['max_time_hours'][rows] = np.where(
        any_valid, info['time_hours'][new_argmax], np.nan)
    info['adjusted'][rows] = True


def resolve_adjacency(prev, cur):
    """
    Enforce the >=24h separation rule between two chronologically adjacent
    months' maxima, mutating `prev`/`cur` in place. Returns the count of
    node-months adjusted by this call (0, 1, or 2 per cascade iteration).
    """
    n_adjusted = 0
    for _ in range(MAX_CASCADE_ITER):
        both_valid = ~np.isnan(prev['max_val']) & ~np.isnan(cur['max_val'])
        delta_hours = np.abs(cur['max_time_hours'] - prev['max_time_hours'])
        violation = both_valid & (delta_hours < MIN_SEPARATION_HOURS)
        if not violation.any():
            break

        prev_smaller = violation & (prev['max_val'] <= cur['max_val'])
        cur_smaller = violation & ~prev_smaller

        if prev_smaller.any():
            _recompute_excluding_argmax(prev, prev_smaller)
            n_adjusted += int(prev_smaller.sum())
        if cur_smaller.any():
            _recompute_excluding_argmax(cur, cur_smaller)
            n_adjusted += int(cur_smaller.sum())

    return n_adjusted


def finalize(info):
    """Drop the heavy raw-data arrays, keeping only the reportable fields."""
    return {
        'year': info['year'], 'month': info['month'],
        'max_val': info['max_val'].astype(np.float32),
        'max_time_hours': info['max_time_hours'],
        'adjusted': info['adjusted'],
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_monthly_max(path, node, metadata, variable_key, months, n_adjusted):
    var_def = nc_metadata.VARIABLES[variable_key]
    n_nodes = len(node['node_index'])
    n_time = len(months)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    ds = nc.Dataset(str(out), 'w', format='NETCDF4')
    nc_metadata.set_global_attrs(
        ds, metadata,
        title=f'Full-period monthly maximum {var_def["long_name"]} '
              f'at SurgeMIP official shoreline',
        summary=(f'Monthly maximum {var_def["long_name"].lower()}, computed '
                 f'from hourly values at the official SurgeMIP shoreline '
                 f'points, over the full simulation period.'),
        timestep='MonthlyMax', variable_key=variable_key,
        feature_type='timeSeries',
        extra={
            'source_csv': node['source_csv'],
            'monthly_max_method': (
                'Calendar-month maximum of the hourly series. Adjacent '
                f'months\' maxima are required to be separated by at least '
                f'{MIN_SEPARATION_HOURS:.0f}h; where violated, the smaller '
                f'of the two maxima is recomputed excluding that timestep. '
                f'See the "{var_def["name"]}_adjusted" variable for which '
                f'node/months were affected.'),
            'monthly_max_min_separation_hours': MIN_SEPARATION_HOURS,
            'n_adjusted_node_months': int(n_adjusted),
        },
    )
    ds.createDimension('node', n_nodes)
    ds.createDimension('time', n_time)

    v = ds.createVariable('time', 'f8', ('time',))
    v.standard_name = 'time'
    v.units = nc_metadata.TIME_UNITS
    v.calendar = 'standard'
    v.axis = 'T'
    v.long_name = 'Start of calendar month'
    v[:] = [nc.date2num(pd.Timestamp(year=m['year'], month=m['month'], day=1)
                        .to_pydatetime(), nc_metadata.TIME_UNITS, 'standard')
            for m in months]

    v = ds.createVariable('year', 'i2', ('time',))
    v.long_name = 'Calendar year'
    v[:] = [m['year'] for m in months]

    v = ds.createVariable('month', 'i2', ('time',))
    v.long_name = 'Calendar month (1-12)'
    v[:] = [m['month'] for m in months]

    max_arr = np.stack([m['max_val'] for m in months], axis=1)  # (node, time)
    max_arr = np.where(np.isnan(max_arr), CF_FILL_F32, max_arr).astype(np.float32)

    v = ds.createVariable(var_def['name'], 'f4', ('node', 'time'),
                          zlib=True, complevel=1,
                          chunksizes=(n_nodes, min(n_time, 120)),
                          fill_value=CF_FILL_F32)
    v.standard_name = var_def['standard_name']
    v.long_name = var_def['long_name']
    v.units = 'm'
    v.cell_methods = 'time: maximum within months'
    v.coordinates = 'node_lon node_lat time'
    v.grid_mapping = 'crs'
    v[:] = max_arr

    time_arr = np.stack([m['max_time_hours'] for m in months], axis=1)
    time_arr = np.where(np.isnan(time_arr), CF_FILL_F32, time_arr)
    v = ds.createVariable(f'{var_def["name"]}_time', 'f8', ('node', 'time'),
                          zlib=True, complevel=1,
                          chunksizes=(n_nodes, min(n_time, 120)),
                          fill_value=CF_FILL_F32)
    v.units = nc_metadata.TIME_UNITS
    v.calendar = 'standard'
    v.long_name = f'Time of the retained monthly maximum {var_def["name"]}'
    v[:] = time_arr

    adj_arr = np.stack([m['adjusted'] for m in months], axis=1).astype('i1')
    v = ds.createVariable(f'{var_def["name"]}_adjusted', 'i1', ('node', 'time'),
                          zlib=True, complevel=1,
                          chunksizes=(n_nodes, min(n_time, 120)))
    v.long_name = (f'1 if this node/month\'s maximum was recomputed due to '
                   f'a <{MIN_SEPARATION_HOURS:.0f}h separation conflict '
                   f'with a neighboring month, else 0')
    v.flag_values = np.array([0, 1], dtype='i1')
    v.flag_meanings = 'unadjusted adjusted'
    v[:] = adj_arr

    crs_v = ds.createVariable('crs', 'i4')
    crs_v.grid_mapping_name = 'latitude_longitude'
    crs_v.longitude_of_prime_meridian = 0.0
    crs_v.semi_major_axis = 6378137.0
    crs_v.inverse_flattening = 298.257223563
    crs_v[:] = -2147483647

    v = ds.createVariable('node_index', 'i4', ('node',), zlib=True, complevel=1)
    v.long_name = 'ADCIRC mesh node index (1-based, matches fort.63.nc node numbering)'
    v.cf_role = 'timeseries_id'
    v[:] = node['node_index'] + 1

    v = ds.createVariable('node_lon', 'f8', ('node',), zlib=True, complevel=1)
    v.standard_name = 'longitude'
    v.units = 'degrees_east'
    v[:] = node['node_lon']

    v = ds.createVariable('node_lat', 'f8', ('node',), zlib=True, complevel=1)
    v.standard_name = 'latitude'
    v.units = 'degrees_north'
    v[:] = node['node_lat']

    v = ds.createVariable('node_depth', 'f8', ('node',), zlib=True, complevel=1)
    v.standard_name = 'sea_floor_depth_below_geoid'
    v.long_name = 'Depth of matched mesh node below geoid'
    v.units = 'm'
    v.positive = 'down'
    v[:] = node['node_depth']

    v = ds.createVariable('point_lon', 'f8', ('node',), zlib=True, complevel=1)
    v.standard_name = 'longitude'
    v.long_name = 'Original CSV point longitude (GSHHS)'
    v.units = 'degrees_east'
    v[:] = node['point_lon']

    v = ds.createVariable('point_lat', 'f8', ('node',), zlib=True, complevel=1)
    v.standard_name = 'latitude'
    v.long_name = 'Original CSV point latitude (GSHHS)'
    v.units = 'degrees_north'
    v[:] = node['point_lat']

    v = ds.createVariable('dist_km', 'f4', ('node',), zlib=True, complevel=1)
    v.long_name = 'Distance from CSV point to matched mesh node'
    v.units = 'km'
    v[:] = node['dist_km'].astype(np.float32)

    nc_metadata.set_geospatial_extent(ds, node['node_lon'], node['node_lat'],
                                      node['node_depth'], positive='down')
    valid_times = [pd.Timestamp(year=m['year'], month=m['month'], day=1)
                   for m in months]
    nc_metadata.update_time_coverage(ds, valid_times)
    ds.close()
    print(f'Wrote {out}  ({n_time} months, {n_adjusted} node-months adjusted '
          f'for the {MIN_SEPARATION_HOURS:.0f}h separation rule)')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    hourly_dir = Path(args.hourly_dir)
    output_dir = Path(args.output_dir)
    var_name = nc_metadata.VARIABLES[args.variable]['name']

    if not hourly_dir.exists():
        print(f'Hourly directory not found: {hourly_dir}')
        sys.exit(1)

    year_files = nc_metadata.discover_hourly_year_files(hourly_dir, var_name)
    if not year_files:
        print(f'No hourly {var_name} files found in {hourly_dir}')
        sys.exit(1)
    print(f'Discovered {len(year_files)} year(s): '
          f'{", ".join(str(y) for y, _ in year_files)}')

    source_attrs = nc_metadata.read_known_attrs(year_files[0][1])
    metadata = nc_metadata.load_metadata(
        args.metadata_yaml, source_attrs=source_attrs,
        cli_overrides=nc_metadata.naming_overrides_from_args(args))

    year_range = f'{year_files[0][0]}-{year_files[-1][0]}'
    out_path = output_dir / nc_metadata.build_filename(
        metadata, 'MonthlyMax', args.variable, year_range)

    if out_path.exists() and not args.force:
        print(f'{out_path} already exists. Use --force to overwrite.')
        return

    node = read_node_metadata(year_files[0][1])

    finalized = []
    total_adjusted = 0
    prev = None
    t0 = timer.time()

    for year, path in year_files:
        print(f'\nReading {path} ...')
        times, data = read_hourly_year(path, var_name)
        time_hours_all = (times - nc_metadata.EPOCH).total_seconds().values / 3600.0

        for month in sorted(times.month.unique()):
            mask = times.month == month
            cur = compute_month_info(year, month, data[:, mask],
                                     time_hours_all[mask])
            if prev is not None:
                total_adjusted += resolve_adjacency(prev, cur)
                finalized.append(finalize(prev))
            prev = cur

        print(f'  {len(finalized)} month(s) finalized so far '
              f'[{timer.time() - t0:.0f}s]')

    if prev is not None:
        finalized.append(finalize(prev))

    print(f'\n{len(finalized)} total month(s), {total_adjusted} node-month(s) '
          f'adjusted for the {MIN_SEPARATION_HOURS:.0f}h separation rule.')

    write_monthly_max(out_path, node, metadata, args.variable, finalized,
                      total_adjusted)


if __name__ == '__main__':
    main()

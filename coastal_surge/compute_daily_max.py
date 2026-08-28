#!/usr/bin/env python3
"""
Compute full-period daily maxima from a directory of per-year hourly
NetCDF files (either twl or ssgh), writing a single output file that spans
the whole record.

This script is Step 3 of the SurgeMIP water level extraction pipeline, and
can be run on the output of either Step 1 (twl) or Step 2 (ssgh):

  Step 1 — extract_outputs_to_shoreline_pts.py   -> hourly twl, per year
  Step 2 — detide_surge.py (optional)            -> hourly ssgh, per year
  Step 3 — this script                           -> full-period DailyMax
  Step 4 — compute_monthly_max.py                -> full-period MonthlyMax

Unlike compute_monthly_max.py, no cross-day adjustment is applied: this is a
plain calendar-day maximum of the hourly series, with no minimum-separation
rule between adjacent days' maxima. Because a calendar day always falls
entirely within a single year's hourly file, each year can be processed
independently in one streaming pass with no state carried across year
boundaries.

Usage:
  python compute_daily_max.py \\
      --hourly-dir /path/to/twl_hourly/ \\
      --variable   WaterLevel \\
      --output-dir /path/to/daily_max/

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
             'compute daily maxima from',
    )
    p.add_argument(
        '--variable', required=True, choices=sorted(nc_metadata.VARIABLES),
        help='Which variable to process: WaterLevel (twl) or StormSurge (ssgh)',
    )
    p.add_argument(
        '--output-dir', required=True,
        help='Directory to write the single full-period DailyMax NetCDF into',
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
    node = nc_metadata.read_node_block(ds)
    ds.close()
    return node


# ---------------------------------------------------------------------------
# Per-day maxima
# ---------------------------------------------------------------------------

def compute_day_max(date, day_data, day_time_hours):
    """
    date : datetime.date
    day_data : (n_nodes, n_hours) float64, CF_FILL_F32 for invalid/dry
    day_time_hours : (n_hours,) float64, hours since nc_metadata.EPOCH

    Returns a finalized dict for this calendar day (no cross-day state).
    """
    n_nodes = day_data.shape[0]
    valid = day_data < (CF_FILL_F32 / 2)
    data = np.where(valid, day_data, -np.inf)

    argmax_idx = np.argmax(data, axis=1)
    any_valid = valid.any(axis=1)
    max_val = data[np.arange(n_nodes), argmax_idx]
    max_val = np.where(any_valid, max_val, np.nan)
    max_time_hours = np.where(any_valid, day_time_hours[argmax_idx], np.nan)

    return {
        'date': date,
        'max_val': max_val.astype(np.float32),
        'max_time_hours': max_time_hours,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_daily_max(path, node, metadata, variable_key, days):
    var_def = nc_metadata.VARIABLES[variable_key]
    n_nodes = len(node['node_index'])
    n_time = len(days)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    point_set_name, point_set_source = nc_metadata.point_set_label(metadata)

    ds = nc.Dataset(str(out), 'w', format='NETCDF4')
    nc_metadata.set_global_attrs(
        ds, metadata,
        title=f'Full-period daily maximum {var_def["long_name"]} '
              f'at the official SurgeMIP {point_set_name}',
        summary=(f'Daily maximum {var_def["long_name"].lower()}, computed '
                 f'from hourly values at the official SurgeMIP '
                 f'{point_set_name}, over the full simulation period.'),
        timestep='DailyMax', variable_key=variable_key,
        feature_type='timeSeries',
        extra={
            'source_csv': node['source_csv'],
            'daily_max_method': 'Calendar-day maximum of the hourly series, '
                                 'with no cross-day adjustment.',
        },
    )
    ds.createDimension('node', n_nodes)
    ds.createDimension('time', n_time)

    v = ds.createVariable('time', 'f8', ('time',))
    v.standard_name = 'time'
    v.units = nc_metadata.TIME_UNITS
    v.calendar = 'standard'
    v.axis = 'T'
    v.long_name = 'Start of calendar day'
    v[:] = [nc.date2num(pd.Timestamp(d['date']).to_pydatetime(),
                        nc_metadata.TIME_UNITS, 'standard')
            for d in days]

    v = ds.createVariable('year', 'i2', ('time',))
    v.long_name = 'Calendar year'
    v[:] = [d['date'].year for d in days]

    v = ds.createVariable('month', 'i2', ('time',))
    v.long_name = 'Calendar month (1-12)'
    v[:] = [d['date'].month for d in days]

    v = ds.createVariable('day', 'i2', ('time',))
    v.long_name = 'Calendar day of month (1-31)'
    v[:] = [d['date'].day for d in days]

    max_arr = np.stack([d['max_val'] for d in days], axis=1)  # (node, time)
    max_arr = np.where(np.isnan(max_arr), CF_FILL_F32, max_arr).astype(np.float32)

    v = ds.createVariable(var_def['name'], 'f4', ('node', 'time'),
                          zlib=True, complevel=1,
                          chunksizes=(n_nodes, min(n_time, 366)),
                          fill_value=CF_FILL_F32)
    v.standard_name = var_def['standard_name']
    v.long_name = var_def['long_name']
    v.units = 'm'
    datum = nc_metadata.resolve_datum(metadata, variable_key)
    if datum:
        v.datum = datum
    v.cell_methods = 'time: maximum within days'
    v.coordinates = 'node_lon node_lat time'
    v.grid_mapping = 'crs'
    v[:] = max_arr

    time_arr = np.stack([d['max_time_hours'] for d in days], axis=1)
    time_arr = np.where(np.isnan(time_arr), CF_FILL_F32, time_arr)
    v = ds.createVariable(f'{var_def["name"]}_time', 'f8', ('node', 'time'),
                          zlib=True, complevel=1,
                          chunksizes=(n_nodes, min(n_time, 366)),
                          fill_value=CF_FILL_F32)
    v.units = nc_metadata.TIME_UNITS
    v.calendar = 'standard'
    v.long_name = f'Time of the retained daily maximum {var_def["name"]}'
    v[:] = time_arr

    nc_metadata.write_node_block(ds, node['model_name'], node,
                                 point_set_source=point_set_source)

    nc_metadata.set_geospatial_extent(
        ds, node['node_lon'], node['node_lat'], node['node_depth'],
        positive='down',
        crs=metadata.get('geospatial_bounds_crs', 'EPSG:4326'),
        vertical_crs=metadata.get('geospatial_bounds_vertical_crs', ''))
    valid_times = [pd.Timestamp(d['date']) for d in days]
    nc_metadata.update_time_coverage(ds, valid_times)
    ds.close()
    print(f'Wrote {out}  ({n_time} days)')


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
        metadata, 'DailyMax', args.variable, year_range)

    if out_path.exists() and not args.force:
        print(f'{out_path} already exists. Use --force to overwrite.')
        return

    node = read_node_metadata(year_files[0][1])

    days = []
    t0 = timer.time()

    for year, path in year_files:
        print(f'\nReading {path} ...')
        times, data = read_hourly_year(path, var_name)
        time_hours_all = (times - nc_metadata.EPOCH).total_seconds().values / 3600.0
        dates = times.date

        for date in sorted(np.unique(dates)):
            mask = dates == date
            days.append(compute_day_max(date, data[:, mask],
                                        time_hours_all[mask]))

        print(f'  {len(days)} day(s) finalized so far '
              f'[{timer.time() - t0:.0f}s]')

    print(f'\n{len(days)} total day(s).')

    write_daily_max(out_path, node, metadata, args.variable, days)


if __name__ == '__main__':
    main()

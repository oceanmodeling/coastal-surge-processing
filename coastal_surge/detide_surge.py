#!/usr/bin/env python3
"""
Fit tidal harmonics across a full multi-year record and write hourly storm
surge height (ssgh) — the non-tidal residual of total water level (twl).

This script is Step 2 (optional) of the SurgeMIP water level extraction
pipeline:

  Step 1 — extract_outputs_to_shoreline_pts.py
      Writes one hourly twl NetCDF file per year.

  Step 2 — this script
      Reads the per-year hourly twl files from Step 1, fits tidal harmonics
      across the full record, subtracts the tidal prediction, and writes one
      hourly ssgh (storm surge height) NetCDF file per year. This step is
      only needed when a StormSurge product is wanted — skip it if only
      total water level is required.

  Step 3 — compute_monthly_max.py
      Computes full-period monthly maxima of either twl or ssgh.

Terminology:
  ssgh (storm_surge_height): the non-tidal residual of twl, i.e. the water
    level with the astronomical tide removed. Preference is to obtain this
    by subtracting an astronomical-tide-only model run from the combined
    astronomical + meteorological run, when one is available. This script
    implements the fallback: harmonic tidal analysis and subtraction, used
    when no astronomical-only run exists.

Tidal model:
  Uses the 67-constituent EXTENDED set from the `detide_extended_constituents`
  submodule (github.com/WPringle/detide, branch adding-extended-constituents),
  MINUS the Sa and Ssa constituents. Sa/Ssa are excluded because they are
  mostly non-astronomical (seasonal/meteorological) in origin, so leaving
  them in the tidal fit would remove real seasonal surge signal rather than
  tide. Nodal corrections are evaluated every PARTITION_HOURS=240h following
  pytides2 convention.

Algorithm (two-phase, restart-safe):

  Phase 1 — Tidal fit (streams through all years once):
      Builds a vectorized linear tidal design matrix A for each year using
      pytides2 (speeds, nodal factors, equilibrium arguments for all 65
      EXTENDED-minus-Sa/Ssa constituents).  Accumulates the normal equations

          A^T A   (n_coefs × n_coefs,  tiny)
          A^T Y   (n_coefs × n_nodes)

      across all years, then solves once for the tidal coefficient matrix

          C = (A^T A)^{-1} A^T Y   (n_coefs × n_nodes)

      Checkpointed after each year so a walltime kill can be resumed without
      losing progress.

  Phase 2 — Subtract tidal prediction (streams through all years once more):
      Re-reads each year's twl file, computes

          ssgh(t) = twl(t) - A(t) @ C

      and writes it to that year's ssgh output file. A year is skipped if
      its output file already exists, unless --force is given.

Usage:
  python detide_surge.py \\
      --hourly-dir  /path/to/twl_hourly/ \\
      --output-dir  /path/to/ssgh_hourly/

Dependencies:
  numpy, netCDF4, pandas,
  pytides2  (pip install git+https://github.com/WPringle/pytides.git@add-tidal-constituents)
  detide    (git submodule: detide_extended_constituents, at the repo root)
"""

import argparse
import sys
import time as timer
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd

import nc_metadata

# Import the tidal constituent list from the SurgeMIP community detide
# library. This library lives in detide_extended_constituents (git
# submodule, at the repo root — sibling of this script's coastal_surge/
# directory), pointing at github.com/WPringle/detide, branch
# adding-extended-constituents.
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / 'detide_extended_constituents'))
from detide.constants import EXTENDED as _EXTENDED_ALL
from pytides2.astro import astro
from pytides2.tide import Tide

# Sa and Ssa are excluded from the detiding constituent set: they are mostly
# non-astronomical (seasonal) in origin, so including them in the harmonic
# fit would strip real seasonal surge signal rather than tide.
EXCLUDED_CONSTITUENTS = ('Sa', 'Ssa')
TIDAL_CONSTITUENTS = [c for c in _EXTENDED_ALL if c.name not in EXCLUDED_CONSTITUENTS]

IN_VARIABLE_KEY  = 'WaterLevel'
OUT_VARIABLE_KEY = 'StormSurge'

PARTITION_HOURS = 240.0  # nodal correction update interval (pytides2 convention)
MIN_PERIODS     = 2      # minimum tidal cycles required to include a constituent

# CF fill value written by extract_outputs_to_shoreline_pts.py for dry/masked nodes
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
        help='Directory of per-year hourly twl NetCDF files written by '
             'extract_outputs_to_shoreline_pts.py',
    )
    p.add_argument(
        '--output-dir', required=True,
        help='Directory to write one hourly ssgh NetCDF file per year into',
    )
    p.add_argument(
        '--force', action='store_true',
        help='Re-extract years whose output file already exists',
    )
    p.add_argument(
        '--metadata-yaml',
        help='Path to a YAML file with global NetCDF metadata. Naming '
             'fields (group_name/climate_forcing/scenario/location) are '
             'otherwise inherited from the input twl files. See '
             'metadata_template.yaml for the editable template.',
    )
    nc_metadata.add_naming_args(p)
    args = p.parse_args()
    return args


# ---------------------------------------------------------------------------
# Input file discovery
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tidal harmonic analysis
# ---------------------------------------------------------------------------

def filter_constituents(constituents, total_hours, t0):
    """Keep only constituents completing at least MIN_PERIODS full cycles."""
    a0 = astro(t0)
    kept = [c for c in constituents
            if 360.0 * MIN_PERIODS < total_hours * c.speed(a0)]
    removed = [c for c in constituents if c not in kept]
    if removed:
        print(f'  Removed (< {MIN_PERIODS} cycles over {total_hours:.0f}h): '
              f'{", ".join(c.name for c in removed)}')
    return kept


def build_tidal_design_matrix(hours, global_t0, constituents, total_hours):
    """
    Build linearised tidal design matrix A for one year.

    The tidal model is:
      h(t) = Z0 + trend*t + sum_i f_i(t)[a_i cos(theta_i(t)) + b_i sin(theta_i(t))]

    where theta_i(t) = speed_i * t + V0_i + u_i(t), with nodal corrections
    f_i and u_i evaluated at the midpoint of each PARTITION_HOURS window.

    Parameters
    ----------
    hours : ndarray (n_times,)
        Hours since global_t0.
    global_t0 : datetime
        Global reference epoch (shared across all years).
    constituents : list
        Tidal constituents from filter_constituents().
    total_hours : float
        Full simulation span in hours (used to normalise the trend column).

    Returns
    -------
    A : ndarray (n_times, 2*n_const + 2)
        Design matrix.  Column 0 = constant; column -1 = normalised trend;
        odd columns 1,3,... = cosine terms; even columns 2,4,... = sine terms.
    """
    n_times = len(hours)
    n_const = len(constituents)

    partitions = Tide._partition(hours, PARTITION_HOURS)
    midpoint_hours = [(p[0] + p[-1]) / 2.0 for p in partitions]
    midpoints = Tide._times(global_t0, midpoint_hours)

    speed, u_list, f_list, V0 = Tide._prepare(
        constituents, global_t0, midpoints, radians=True)

    n_cols = 2 * n_const + 2
    A = np.ones((n_times, n_cols), dtype=np.float64)

    offset = 0
    for part_hours, u_i, f_i in zip(partitions, u_list, f_list):
        n_p = len(part_hours)
        t_col = part_hours[:, np.newaxis]
        arg   = speed.T * t_col + (V0 + u_i).T
        f_row = f_i.T
        A[offset:offset + n_p, 1:-1:2] = f_row * np.cos(arg)
        A[offset:offset + n_p, 2:-1:2] = f_row * np.sin(arg)
        offset += n_p

    all_hours = np.concatenate(partitions)
    A[:, -1] = all_hours / total_hours  # normalised trend
    return A


# ---------------------------------------------------------------------------
# Phase 1 checkpoint helpers
# ---------------------------------------------------------------------------

def checkpoint_path(output_dir):
    return Path(output_dir) / '.detide_tidal_checkpoint.npz'


def save_checkpoint(ckpt_path, ATA, ATY, global_t0_iso,
                    accumulated_years, constituent_names):
    np.savez(str(ckpt_path),
             ATA=ATA, ATY=ATY,
             global_t0_iso=global_t0_iso,
             accumulated_years=np.array(sorted(accumulated_years)),
             constituent_names=np.array(constituent_names))
    print(f'  Checkpoint saved ({len(accumulated_years)} years accumulated)')


def load_checkpoint(ckpt_path):
    data = np.load(str(ckpt_path), allow_pickle=True)
    return {
        'ATA':               data['ATA'],
        'ATY':               data['ATY'],
        'global_t0_iso':     str(data['global_t0_iso']),
        'accumulated_years': set(int(y) for y in data['accumulated_years']),
        'constituent_names': list(data['constituent_names']),
    }


# ---------------------------------------------------------------------------
# Per-year file I/O
# ---------------------------------------------------------------------------

def read_hourly_year(path, var_name):
    """Read one year's hourly file in full: times, data (n_nodes, n_times)."""
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


def _write_node_metadata(ds, node):
    n_nodes = len(node['node_index'])

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

    for name, key, std_name, long_name, units in [
        ('node_lon', 'node_lon', 'longitude',
         'Longitude of matched ADCIRC mesh node', 'degrees_east'),
        ('node_lat', 'node_lat', 'latitude',
         'Latitude of matched ADCIRC mesh node', 'degrees_north'),
        ('node_depth', 'node_depth', 'sea_floor_depth_below_geoid',
         'Depth of matched mesh node below geoid', 'm'),
        ('point_lon', 'point_lon', 'longitude',
         'Original CSV point longitude (GSHHS)', 'degrees_east'),
        ('point_lat', 'point_lat', 'latitude',
         'Original CSV point latitude (GSHHS)', 'degrees_north'),
    ]:
        v = ds.createVariable(name, 'f8', ('node',), zlib=True, complevel=1)
        v.standard_name = std_name
        v.long_name = long_name
        v.units = units
        v[:] = node[key]
        if name == 'node_depth':
            v.positive = 'down'

    v = ds.createVariable('dist_km', 'f4', ('node',), zlib=True, complevel=1)
    v.long_name = 'Distance from CSV point to matched mesh node'
    v.units = 'km'
    v[:] = node['dist_km'].astype(np.float32)


def write_ssgh_year(path, node, metadata, times, surge_data, constituents):
    """Write one year of hourly ssgh data (n_nodes, n_times) to `path`."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    var_def = nc_metadata.VARIABLES[OUT_VARIABLE_KEY]
    n_nodes = surge_data.shape[0]
    n_times = surge_data.shape[1]

    ds = nc.Dataset(str(out), 'w', format='NETCDF4')
    nc_metadata.set_global_attrs(
        ds, metadata,
        title=f'Hourly {var_def["long_name"]} at SurgeMIP official shoreline',
        summary=(f'Hourly {var_def["long_name"].lower()}, the non-tidal '
                 f'residual of total water level, computed by subtracting '
                 f'a least-squares tidal harmonic fit from hourly ADCIRC '
                 f'water levels at the official SurgeMIP shoreline points.'),
        feature_type='timeSeries',
        extra={
            'source_csv': node['source_csv'],
            'detiding_method': (
                'Vectorized linear least-squares harmonic analysis across '
                'all years. Tidal coefficients: C = (A^T A)^{-1} A^T Y. '
                f'Nodal corrections every {PARTITION_HOURS}h via pytides2.'),
            'detiding_constituents': ', '.join(c.name for c in constituents),
            'detiding_excluded_constituents': ', '.join(EXCLUDED_CONSTITUENTS),
        },
    )
    ds.createDimension('node', n_nodes)
    ds.createDimension('time', n_times)

    v = ds.createVariable('time', 'f8', ('time',))
    v.standard_name = 'time'
    v.units = nc_metadata.TIME_UNITS
    v.calendar = 'standard'
    v.axis = 'T'
    v[:] = nc_metadata.write_times(ds, 'time', times)

    v = ds.createVariable(var_def['name'], 'f4', ('node', 'time'),
                          zlib=True, complevel=1,
                          chunksizes=(n_nodes, min(n_times, 24)),
                          fill_value=CF_FILL_F32)
    v.standard_name = var_def['standard_name']
    v.long_name = var_def['long_name']
    v.units = 'm'
    v.coordinates = 'node_lon node_lat time'
    v.grid_mapping = 'crs'

    chunk = 10000
    for i in range(0, n_nodes, chunk):
        j = min(i + chunk, n_nodes)
        v[i:j, :] = surge_data[i:j, :]

    _write_node_metadata(ds, node)
    nc_metadata.set_geospatial_extent(ds, node['node_lon'], node['node_lat'],
                                      node['node_depth'], positive='down')
    nc_metadata.update_time_coverage(ds, times)
    ds.close()
    print(f'Wrote {out}')


# ---------------------------------------------------------------------------
# Phase 1: accumulate normal equations and solve for tidal coefficients
# ---------------------------------------------------------------------------

def run_phase1(year_files, output_dir):
    """
    Stream through the per-year twl files once to fit tidal harmonics.

    Returns (C, global_t0, constituents, total_hours) where
    C is shape (n_coefs, n_nodes).
    """
    ckpt_path = checkpoint_path(output_dir)
    all_years = [y for y, _ in year_files]

    first_times, _ = read_hourly_year(year_files[0][1],
                                      nc_metadata.VARIABLES[IN_VARIABLE_KEY]['name'])
    last_times, _  = read_hourly_year(year_files[-1][1],
                                      nc_metadata.VARIABLES[IN_VARIABLE_KEY]['name'])

    global_t0     = first_times[0].to_pydatetime()
    global_t0_iso = global_t0.isoformat()
    total_hours   = (last_times[-1] - first_times[0]).total_seconds() / 3600.0

    print(f'Global reference time: {global_t0_iso}')
    print(f'Total time span: {total_hours:.0f}h ({total_hours/8760:.1f} yr)')

    constituents       = filter_constituents(TIDAL_CONSTITUENTS, total_hours,
                                             global_t0)
    n_coefs            = 2 * len(constituents) + 2
    constituent_names  = [c.name for c in constituents]
    print(f'{len(constituents)} tidal constituents (Sa/Ssa excluded): '
          f'{", ".join(constituent_names)}')

    n_nodes = None
    ATA = np.zeros((n_coefs, n_coefs), dtype=np.float64)
    ATY = None
    accumulated_years = set()

    if ckpt_path.exists():
        ckpt = load_checkpoint(ckpt_path)
        if (ckpt['global_t0_iso'] == global_t0_iso and
                list(ckpt['constituent_names']) == constituent_names):
            ATA, ATY = ckpt['ATA'], ckpt['ATY']
            accumulated_years = ckpt['accumulated_years']
            n_nodes = ATY.shape[1]
            print(f'Resumed from checkpoint: '
                  f'{len(accumulated_years)} years already accumulated')
        else:
            print('Checkpoint incompatible (different t0 or constituents), '
                  'starting fresh')

    phase1_t0 = timer.time()
    for i, (year, path) in enumerate(year_files):
        if year in accumulated_years:
            print(f'  Year {year}: already accumulated, skipping')
            continue

        print(f'\n--- Phase 1: Year {year} [{i+1}/{len(year_files)}] ---')
        t0_wall = timer.time()

        times, data = read_hourly_year(
            path, nc_metadata.VARIABLES[IN_VARIABLE_KEY]['name'])
        if n_nodes is None:
            n_nodes = data.shape[0]
            ATY = np.zeros((n_coefs, n_nodes), dtype=np.float64)

        hours = (times - pd.Timestamp(global_t0)).total_seconds().values / 3600.0
        A     = build_tidal_design_matrix(hours, global_t0, constituents,
                                          total_hours)

        vals  = data.T                             # (n_times, n_nodes)
        valid = vals < CF_FILL_F32 / 2
        vals  = np.where(valid, vals, 0.0)          # zero-out dry sentinels

        ATA += A.T @ A
        ATY += A.T @ vals

        accumulated_years.add(year)
        save_checkpoint(ckpt_path, ATA, ATY, global_t0_iso,
                        accumulated_years, constituent_names)
        print(f'  Accumulated in {timer.time()-t0_wall:.1f}s')

    print(f'\nPhase 1 complete in {timer.time()-phase1_t0:.1f}s')
    print('Solving tidal coefficients ...')
    C = np.linalg.solve(ATA, ATY)
    print(f'  C shape: {C.shape}')
    return C, global_t0, constituents, total_hours


# ---------------------------------------------------------------------------
# Phase 2: subtract tidal prediction, write hourly ssgh per year
# ---------------------------------------------------------------------------

def run_phase2(year_files, C, global_t0, constituents, total_hours,
              output_dir, metadata, force):
    output_dir = Path(output_dir)

    todo = []
    for year, in_path in year_files:
        out_path = output_dir / nc_metadata.build_filename(
            metadata, 'Hourly', OUT_VARIABLE_KEY, year)
        if out_path.exists() and not force:
            continue
        todo.append((year, in_path, out_path))

    skipped = len(year_files) - len(todo)
    if skipped:
        print(f'Skipping {skipped} already-extracted year(s) '
              f'(use --force to redo).')

    if not todo:
        print('Nothing to extract in Phase 2.')
        return

    total_t0 = timer.time()
    for i, (year, in_path, out_path) in enumerate(todo):
        print(f'\n{"="*60}')
        print(f'Phase 2: Year {year} [{i+1}/{len(todo)}]')
        print(f'{"="*60}')

        times, data = read_hourly_year(
            in_path, nc_metadata.VARIABLES[IN_VARIABLE_KEY]['name'])
        node = read_node_metadata(in_path)

        hours = (times - pd.Timestamp(global_t0)).total_seconds().values / 3600.0
        A     = build_tidal_design_matrix(hours, global_t0, constituents,
                                          total_hours)

        vals  = data.T                              # (n_times, n_nodes)
        valid = vals < CF_FILL_F32 / 2
        tide  = A @ C                                # (n_times, n_nodes)
        surge = np.where(valid, vals - tide, CF_FILL_F32)
        surge_data = surge.T.astype(np.float32)      # (n_nodes, n_times)

        write_ssgh_year(out_path, node, metadata, times, surge_data,
                        constituents)

        elapsed = timer.time() - total_t0
        print(f'  Year {year} done. Cumulative: {elapsed:.0f}s')

    print(f'\nPhase 2 total: {timer.time()-total_t0:.1f}s')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    hourly_dir = Path(args.hourly_dir)
    output_dir = Path(args.output_dir)

    if not hourly_dir.exists():
        print(f'Hourly directory not found: {hourly_dir}')
        sys.exit(1)

    in_var_name = nc_metadata.VARIABLES[IN_VARIABLE_KEY]['name']
    year_files = nc_metadata.discover_hourly_year_files(hourly_dir, in_var_name)
    if not year_files:
        print(f'No hourly {in_var_name} files found in {hourly_dir}')
        sys.exit(1)
    print(f'Discovered {len(year_files)} year(s): '
          f'{", ".join(str(y) for y, _ in year_files)}')

    # Inherit naming/provenance metadata from the twl input files (they were
    # written with group_name/climate_forcing/scenario/location already
    # set), unless overridden by --metadata-yaml or CLI flags.
    source_attrs = nc_metadata.read_known_attrs(year_files[0][1])
    metadata = nc_metadata.load_metadata(
        args.metadata_yaml, source_attrs=source_attrs,
        cli_overrides=nc_metadata.naming_overrides_from_args(args))

    # Fail fast if the naming fields aren't resolvable, before the expensive
    # tidal fit.
    nc_metadata.build_filename(metadata, 'Hourly', OUT_VARIABLE_KEY,
                               year_files[0][0])

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"="*60}')
    print('PHASE 1: Tidal harmonic analysis')
    print(f'{"="*60}')
    C, global_t0, constituents, total_hours = run_phase1(year_files, output_dir)

    print(f'\n{"="*60}')
    print('PHASE 2: Subtract tidal prediction, write hourly ssgh')
    print(f'{"="*60}')
    run_phase2(year_files, C, global_t0, constituents, total_hours,
              output_dir, metadata, args.force)

    ckpt = checkpoint_path(output_dir)
    if ckpt.exists():
        ckpt.unlink()
        print(f'Removed checkpoint: {ckpt}')

    print('\nDone.')


if __name__ == '__main__':
    main()

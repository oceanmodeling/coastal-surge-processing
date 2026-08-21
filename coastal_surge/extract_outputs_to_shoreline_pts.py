#!/usr/bin/env python3
"""
Extract hourly total water level (twl) at the official SurgeMIP shoreline
points from ADCIRC fort.63.nc files.

Each lon/lat point in the input CSV is matched to the nearest node in the
full 13.4M-node ADCIRC mesh via an ECEF KDTree.  The extracted time series
are written one compact CF-1.8 compliant NetCDF file per calendar year
(~1.2 GB/year at float32 for 35k nodes, versus ~550 GB/year for the raw
fort.63.nc), following the SurgeMIP naming convention:

    twl_Hourly_GroupName_ClimateForcing_Scenario_Location_TimeRange.nc

This script is Step 1 of the SurgeMIP water level extraction pipeline:

  Step 1 — this script
      Reads raw ADCIRC fort.63.nc files (one per year), extracts hourly
      total water level at the official SurgeMIP shoreline points, and
      writes one compact per-year CF-1.8 NetCDF file.

  Step 2 — detide_surge.py (optional)
      Fits tidal harmonics across the per-year hourly twl files from Step 1
      and writes per-year hourly storm surge height (ssgh) files.

  Step 3 — compute_monthly_max.py
      Computes full-period monthly maxima (of either twl or ssgh) from a
      directory of per-year hourly files.

Terminology:
  twl (total_water_level): astronomical + meteorological driven water level
    (e.g. storm tide); may include other contributions depending on the
    model. See the "source"/"forcing"/"comment" global attributes for
    model-specific details.

Algorithm (per year, restart-safe — a year is skipped if its output file
already exists, unless --force is given):
  1. Read fort.63.nc in batches of --batch-size full rows (sequential I/O).
  2. Select the matched nodes from each batch and convert to float32.
  3. Accumulate the full year's data in memory (~1.2 GB at float32).
  4. Write the whole year to its own output file.

Usage:
  python extract_outputs_to_shoreline_pts.py \\
      --points-csv coastal_points_gsshs_low_20km_35k-pts.csv \\
      --adcirc-dir ${OUTPUT_DIR}/CFS-reanalysis/ \\
      --output-dir shoreline_extremes/ \\
      --metadata-yaml coastal_surge/metadata_template.yaml \\
      --group-name Argonne --climate-forcing CFSv2 --scenario Reanalysis \\
      --location GESLA

Dependencies:
  numpy, netCDF4, pandas, scipy
"""

import argparse
import re
import sys
import time as timer
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd

import nc_metadata

VARIABLE_KEY = 'WaterLevel'

# CF standard fill value for float32
CF_FILL_F32 = 9.96921e+36


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        '--points-csv', required=True,
        help='CSV with Longitude,Latitude columns (e.g. SurgeMIP 35k points)',
    )
    adcirc_group = p.add_mutually_exclusive_group(required=True)
    adcirc_group.add_argument(
        '--adcirc-dir',
        help='Directory containing year subdirectories with fort.63.nc files',
    )
    adcirc_group.add_argument(
        '--adcirc', nargs='+',
        help='Specific fort.63.nc file(s) to process',
    )
    p.add_argument(
        '--output-dir', required=True,
        help='Directory to write one hourly twl NetCDF file per year into, '
             'named per the SurgeMIP convention (see module docstring).',
    )
    p.add_argument(
        '--force', action='store_true',
        help='Re-extract years whose output file already exists',
    )
    p.add_argument(
        '--batch-size', type=int, default=72,
        help='Number of timesteps to read per slab (default: 72). '
             'Larger values use more memory but fewer Lustre reads. '
             'Memory per batch: batch_size × 13.4M × 8 bytes.',
    )
    p.add_argument(
        '--wet-mask',
        help='Path to always_wet_mask.npz produced by build_always_wet_mask.py. '
             'When supplied, the KDTree is restricted to always-wet nodes only, '
             'preventing CSV points from being matched to intermittently-dry '
             'floodplain nodes.',
    )
    p.add_argument(
        '--metadata-yaml',
        help='Path to a YAML file with global NetCDF metadata (institution, '
             'contact, project, license, naming fields, ...). See '
             'metadata_template.yaml for the editable template. Fields not '
             'present fall back to built-in defaults.',
    )
    nc_metadata.add_naming_args(p)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Fort.63.nc discovery
# ---------------------------------------------------------------------------

def find_best_fort63(year_dir):
    year_dir = Path(year_dir)
    pattern = re.compile(r'^fort\.63\.nc(-\d+)?$')
    valid = []
    for c in sorted(year_dir.glob('fort.63.nc*')):
        if pattern.match(c.name):
            m = re.search(r'-(\d+)$', c.name)
            suffix = int(m.group(1)) if m else 0
            valid.append((suffix, c))
    if not valid:
        return None
    valid.sort(key=lambda x: x[0])
    return valid[-1][1]


def discover_adcirc_files(adcirc_dir):
    adcirc_dir = Path(adcirc_dir)
    results = []
    for subdir in sorted(adcirc_dir.iterdir()):
        if not subdir.is_dir():
            continue
        if not re.match(r'^\d{4}$', subdir.name):
            continue
        best = find_best_fort63(subdir)
        if best is not None:
            results.append((int(subdir.name), best))
    results.sort(key=lambda x: x[0])
    return results


def get_adcirc_year(adcirc_path):
    ds = nc.Dataset(str(adcirc_path), 'r')
    cal = getattr(ds.variables['time'], 'calendar', 'standard')
    t0 = nc.num2date(ds.variables['time'][0], ds.variables['time'].units, cal)
    ds.close()
    return t0.year


# ---------------------------------------------------------------------------
# Official point matching (full-mesh KDTree)
# ---------------------------------------------------------------------------

def load_official_points(csv_path, adcirc_path, wet_mask_path=None):
    """
    Match lon/lat points from a CSV to the nearest node in the full ADCIRC mesh.

    Reads the full 13.4M-node mesh from fort.63.nc (x, y, depth variables)
    and builds an ECEF KDTree to find the nearest mesh node for each CSV point.
    No maximum-distance cutoff is applied (max_dist_km=1e6); all points will
    match.  A warning is printed for any match exceeding 100 km.

    When wet_mask_path is supplied, the KDTree is built from always-wet nodes
    only (loaded from the .npz produced by build_always_wet_mask.py).  The
    returned node_index values are still global 0-based indices into the full
    mesh, so all downstream slab reads remain correct.

    Multiple CSV points may map to the same mesh node when the CSV spacing
    exceeds the local mesh resolution.  Duplicates are preserved to maintain
    1:1 correspondence between CSV rows and output nodes.

    Parameters
    ----------
    csv_path : str or Path
    adcirc_path : str or Path
        Any fort.63.nc from the campaign (mesh geometry is static).
    wet_mask_path : str or Path or None
        Path to always_wet_mask.npz.  If None, the full mesh is searched.

    Returns
    -------
    node_index : ndarray int64 (n_points,)   0-based mesh node indices
    node_lon   : ndarray float64 (n_points,) matched mesh node longitude
    node_lat   : ndarray float64 (n_points,) matched mesh node latitude
    node_depth : ndarray float64 (n_points,) matched mesh node depth
    point_lon  : ndarray float64 (n_points,) original CSV longitude
    point_lat  : ndarray float64 (n_points,) original CSV latitude
    dist_km    : ndarray float64 (n_points,) match distance in km
    """
    from surgemip_stations import find_nearest_nodes

    print(f'Loading official points from {csv_path} ...')
    df = pd.read_csv(str(csv_path))
    point_lon = df['Longitude'].values.astype(np.float64)
    point_lat = df['Latitude'].values.astype(np.float64)
    print(f'  {len(point_lon):,} points')

    print(f'Loading full mesh from {adcirc_path} ...')
    t0 = timer.time()
    ds = nc.Dataset(str(adcirc_path), 'r')
    mesh_lon = np.array(ds.variables['x'][:], dtype=np.float64)
    mesh_lat = np.array(ds.variables['y'][:], dtype=np.float64)
    mesh_depth = np.array(ds.variables['depth'][:], dtype=np.float64)
    ds.close()
    print(f'  {len(mesh_lon):,} mesh nodes loaded in {timer.time() - t0:.1f}s')

    # Optionally restrict the search to always-wet nodes
    if wet_mask_path is not None:
        print(f'Loading always-wet mask from {wet_mask_path} ...')
        data = np.load(str(wet_mask_path))
        always_wet = data['always_wet']
        if len(always_wet) != len(mesh_lon):
            print(f'ERROR: wet mask has {len(always_wet)} entries but mesh has '
                  f'{len(mesh_lon)} nodes — mask was built from a different mesh.')
            sys.exit(1)
        wet_indices = np.where(always_wet)[0]  # global indices of wet nodes
        n_wet = len(wet_indices)
        print(f'  {n_wet:,} / {len(mesh_lon):,} nodes are always wet '
              f'({100 * n_wet / len(mesh_lon):.1f}%)')
        # KDTree over wet-node-only sub-mesh; find_nearest_nodes returns
        # indices into the sub-array, which we then map back to global indices
        sub_lon = mesh_lon[wet_indices]
        sub_lat = mesh_lat[wet_indices]
        sub_indices, dist_km = find_nearest_nodes(
            point_lon, point_lat, sub_lon, sub_lat, max_dist_km=1e6)
        # sub_indices are positions in wet_indices; map back to global mesh
        node_indices = wet_indices[sub_indices].astype(np.int64)
    else:
        node_indices, dist_km = find_nearest_nodes(
            point_lon, point_lat, mesh_lon, mesh_lat, max_dist_km=1e6)
        node_indices = node_indices.astype(np.int64)

    far = dist_km > 100.0
    if far.any():
        print(f'  WARNING: {far.sum()} point(s) are >100 km from nearest mesh '
              f'node (max {dist_km.max():.1f} km). Check CSV coordinates.')

    print(f'  Match distance: median={np.median(dist_km):.2f} km, '
          f'max={dist_km.max():.2f} km')

    return (node_indices,
            mesh_lon[node_indices],
            mesh_lat[node_indices],
            mesh_depth[node_indices],
            point_lon,
            point_lat,
            dist_km)


# ---------------------------------------------------------------------------
# Output file writing (one file per year, written in a single shot)
# ---------------------------------------------------------------------------

def _write_node_metadata(ds, node_index, node_lon, node_lat, node_depth,
                          point_lon, point_lat, dist_km):
    """Write static node-dimension variables."""
    n_nodes = len(node_index)

    # CRS scalar (CF convention for geographic CRS)
    crs_v = ds.createVariable('crs', 'i4')
    crs_v.grid_mapping_name = 'latitude_longitude'
    crs_v.longitude_of_prime_meridian = 0.0
    crs_v.semi_major_axis = 6378137.0
    crs_v.inverse_flattening = 298.257223563
    crs_v[:] = -2147483647  # conventional dummy value

    v = ds.createVariable('node_index', 'i4', ('node',),
                          zlib=True, complevel=1)
    v.long_name = 'ADCIRC mesh node index (1-based, matches fort.63.nc node numbering)'
    v.cf_role = 'timeseries_id'
    v[:] = node_index + 1

    v = ds.createVariable('node_lon', 'f8', ('node',),
                          zlib=True, complevel=1)
    v.standard_name = 'longitude'
    v.long_name = 'Longitude of matched ADCIRC mesh node'
    v.units = 'degrees_east'
    v[:] = node_lon

    v = ds.createVariable('node_lat', 'f8', ('node',),
                          zlib=True, complevel=1)
    v.standard_name = 'latitude'
    v.long_name = 'Latitude of matched ADCIRC mesh node'
    v.units = 'degrees_north'
    v[:] = node_lat

    v = ds.createVariable('node_depth', 'f8', ('node',),
                          zlib=True, complevel=1)
    v.standard_name = 'sea_floor_depth_below_geoid'
    v.long_name = 'Depth of matched mesh node below geoid'
    v.units = 'm'
    v.positive = 'down'
    v[:] = node_depth

    v = ds.createVariable('point_lon', 'f8', ('node',),
                          zlib=True, complevel=1)
    v.standard_name = 'longitude'
    v.long_name = 'Original CSV point longitude (GSHHS)'
    v.units = 'degrees_east'
    v[:] = point_lon

    v = ds.createVariable('point_lat', 'f8', ('node',),
                          zlib=True, complevel=1)
    v.standard_name = 'latitude'
    v.long_name = 'Original CSV point latitude (GSHHS)'
    v.units = 'degrees_north'
    v[:] = point_lat

    v = ds.createVariable('dist_km', 'f4', ('node',),
                          zlib=True, complevel=1)
    v.long_name = 'Distance from CSV point to matched mesh node'
    v.units = 'km'
    v[:] = dist_km.astype(np.float32)


def write_hourly_year(path, n_nodes, node_index, node_lon, node_lat,
                       node_depth, point_lon, point_lat, dist_km, csv_name,
                       metadata, times, year_data):
    """
    Create and fully write one year's hourly twl NetCDF file in one shot.

    year_data : float32 ndarray (n_nodes, n_times)
    times : pd.DatetimeIndex (n_times,)
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    var_def = nc_metadata.VARIABLES[VARIABLE_KEY]
    n_times = len(times)

    ds = nc.Dataset(str(out), 'w', format='NETCDF4')
    nc_metadata.set_global_attrs(
        ds, metadata,
        title=f'Hourly {var_def["long_name"]} at SurgeMIP official shoreline',
        summary=(f'Hourly {var_def["long_name"].lower()} extracted from '
                 f'ADCIRC fort.63.nc output at the official SurgeMIP '
                 f'shoreline points.'),
        feature_type='timeSeries',
        extra={'source_csv': csv_name},
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
        v[i:j, :] = year_data[i:j, :]

    _write_node_metadata(ds, node_index, node_lon, node_lat, node_depth,
                         point_lon, point_lat, dist_km)
    nc_metadata.set_geospatial_extent(ds, node_lon, node_lat, node_depth,
                                      positive='down')
    nc_metadata.update_time_coverage(ds, times)
    ds.close()
    print(f'Wrote {out}')


# ---------------------------------------------------------------------------
# Year extraction
# ---------------------------------------------------------------------------

def read_time_axis(adcirc_path):
    ds = nc.Dataset(str(adcirc_path), 'r')
    cal = getattr(ds.variables['time'], 'calendar', 'standard')
    cftime_dates = nc.num2date(ds.variables['time'][:],
                               ds.variables['time'].units, cal)
    times = pd.to_datetime([d.isoformat() for d in cftime_dates])
    ds.close()
    return times


def extract_year(adcirc_path, sorted_nodes, unsort, batch_size):
    """
    Read one year's fort.63.nc and return the full time series at sorted_nodes.

    Parameters
    ----------
    adcirc_path : Path
    sorted_nodes : ndarray int, sorted node indices for efficient slab selection
    unsort : ndarray int, argsort(sort_order) to restore CSV row order
    batch_size : int

    Returns
    -------
    times : pd.DatetimeIndex, shape (n_times,)
    year_data : ndarray float32, shape (n_points, n_times), CSV row order
    """
    ds = nc.Dataset(str(adcirc_path), 'r')
    cal = getattr(ds.variables['time'], 'calendar', 'standard')
    cftime_dates = nc.num2date(ds.variables['time'][:],
                               ds.variables['time'].units, cal)
    times = pd.to_datetime([d.isoformat() for d in cftime_dates])

    n_times = len(times)
    n_nodes = len(sorted_nodes)
    zeta_var = ds.variables['zeta']

    year_data = np.full((n_nodes, n_times), CF_FILL_F32, dtype=np.float32)

    t0_wall = timer.time()
    print(f'  {n_times:,} timesteps, batch_size={batch_size}')

    for t_s in range(0, n_times, batch_size):
        t_e = min(t_s + batch_size, n_times)

        slab = zeta_var[t_s:t_e, :]                           # [B, 13.4M] float64
        vals = np.array(slab[:, sorted_nodes], dtype=np.float32)  # [B, n_nodes]
        del slab

        # Replace ADCIRC dry-node sentinel with CF fill
        vals[vals < -9999] = CF_FILL_F32
        year_data[:, t_s:t_e] = vals.T                        # [n_nodes, B]

        if t_e % (batch_size * 20) < batch_size or t_e == n_times:
            elapsed = timer.time() - t0_wall
            print(f'    [{t_e / n_times * 100:.0f}%] {t_e}/{n_times} '
                  f'[{elapsed:.0f}s]', flush=True)

    ds.close()
    elapsed = timer.time() - t0_wall
    print(f'  Read {n_times} timesteps in {elapsed:.1f}s '
          f'({elapsed / n_times * 1000:.1f}ms/step)')

    # Restore CSV row order
    return times, year_data[unsort, :]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    csv_name = Path(args.points_csv).name

    if args.batch_size < 1:
        print('ERROR: --batch-size must be >= 1')
        sys.exit(1)

    # --- Discover fort.63.nc files ---
    if args.adcirc_dir:
        file_list = discover_adcirc_files(args.adcirc_dir)
        if not file_list:
            print(f'No fort.63.nc files found in {args.adcirc_dir}')
            sys.exit(1)
        print(f'Discovered {len(file_list)} year(s): '
              f'{", ".join(str(y) for y, _ in file_list)}')
    else:
        file_list = []
        for path_str in args.adcirc:
            p = Path(path_str)
            if not p.exists():
                print(f'File not found: {p}')
                sys.exit(1)
            file_list.append((get_adcirc_year(p), p))
        file_list.sort(key=lambda x: x[0])

    # Inherit any matching global attrs already present on the source
    # fort.63.nc (e.g. institution/source set by the ADCIRC run) when no
    # --metadata-yaml is given, or to fill gaps left by one.
    source_attrs = nc_metadata.read_known_attrs(file_list[0][1])
    metadata = nc_metadata.load_metadata(
        args.metadata_yaml, source_attrs=source_attrs,
        cli_overrides=nc_metadata.naming_overrides_from_args(args))

    # Fail fast if the naming fields aren't resolvable, before any expensive
    # extraction work.
    nc_metadata.build_filename(metadata, 'Hourly', VARIABLE_KEY,
                               file_list[0][0])

    # --- Match official points to mesh (uses first fort.63.nc for geometry) ---
    (node_index, node_lon, node_lat, node_depth,
     point_lon, point_lat, dist_km) = load_official_points(
        args.points_csv, file_list[0][1],
        wet_mask_path=args.wet_mask)

    n_nodes = len(node_index)

    # Sort node_index for fast slab selection; unsort restores CSV row order
    sort_order = np.argsort(node_index)
    sorted_nodes = node_index[sort_order]
    unsort = np.argsort(sort_order)

    # --- Determine which years to extract ---
    todo = []
    for year, adcirc_path in file_list:
        out_path = out_dir / nc_metadata.build_filename(
            metadata, 'Hourly', VARIABLE_KEY, year)
        if out_path.exists() and not args.force:
            continue
        todo.append((year, adcirc_path, out_path))

    skipped = len(file_list) - len(todo)
    if skipped:
        print(f'Skipping {skipped} already-extracted year(s) '
              f'(use --force to redo).')

    if not todo:
        print('All years already extracted. Use --force to redo.')
        return

    # --- Extract each year ---
    total_t0 = timer.time()
    for i, (year, adcirc_path, out_path) in enumerate(todo):
        print(f'\n{"=" * 60}')
        print(f'Year {year}: {adcirc_path}  [{i + 1}/{len(todo)}]')
        print(f'{"=" * 60}')

        times, year_data = extract_year(
            adcirc_path, sorted_nodes, unsort, args.batch_size)

        write_hourly_year(
            out_path, n_nodes, node_index, node_lon, node_lat, node_depth,
            point_lon, point_lat, dist_km, csv_name, metadata, times,
            year_data)

        elapsed = timer.time() - total_t0
        print(f'  Year {year} done. Cumulative: {elapsed:.0f}s')

    total_elapsed = timer.time() - total_t0
    print(f'\nAll done. Total: {total_elapsed:.1f}s '
          f'({total_elapsed / 3600:.2f}h)')
    print(f'  Output directory: {out_dir}')


if __name__ == '__main__':
    main()

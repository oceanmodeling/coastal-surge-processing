#!/usr/bin/env python3
"""
Extract hourly water surface elevations and monthly maxima at the official
SurgeMIP shoreline points from ADCIRC fort.63.nc files.

Each lon/lat point in the input CSV is matched to the nearest node in the
full 13.4M-node ADCIRC mesh via an ECEF KDTree.  The extracted time series
are written to a compact CF-1.8 compliant NetCDF (~55 GB for 46 years at
float32, versus 550 GB/year for the raw fort.63.nc).  Monthly maxima are
computed on-the-fly from the in-memory year buffer and written to a
separate lightweight file.

This script is the primary I/O step for the SurgeMIP water level extraction
pipeline.  All downstream analysis (tidal harmonic fitting, surge extraction,
GPD return-level estimation) should operate on the compact output files rather
than re-reading fort.63.nc.

Algorithm (per year, restart-safe):
  1. Read fort.63.nc in batches of --batch-size full rows (sequential I/O).
  2. Select the 35k matched nodes from each batch and convert to float32.
  3. Accumulate the full year's data in memory (~1.2 GB at float32).
  4. Write hourly data to the unlimited-time output file.
  5. Compute monthly maxima from the in-memory buffer and write to a second
     file.  No second read of fort.63.nc required.

Output:
  --output-hourly   CF-1.8 NetCDF, dimensions (node, time), zlib compressed.
                    Chunked (n_nodes, 1): one chunk = one timestep row, fast
                    for downstream tidal fitting which reads row by row.
  --output-monthly  CF-1.8 NetCDF, dimensions (node, time) where time is a
                    monthly record.  ~78 MB for 46 years.

Usage:
  python extract_outputs_to_shoreline_pts.py \\
      --points-csv coastal_points_gsshs_low_20km_35k-pts.csv \\
      --adcirc-dir ${OUTPUT_DIR}/CFS-reanalysis/ \\
      --output-hourly  shoreline_extremes/cfs_reanalysis_35k_hourly.nc \\
      --output-monthly shoreline_extremes/cfs_reanalysis_35k_monthly_max.nc

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
        '--output-hourly', required=True,
        help='Output NetCDF path for hourly water levels',
    )
    p.add_argument(
        '--output-monthly', required=True,
        help='Output NetCDF path for monthly maxima',
    )
    p.add_argument(
        '--force', action='store_true',
        help='Re-extract years already present in output files',
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
             'contact, project, license, ...). See metadata_template.yaml '
             'for the editable template. Fields not present fall back to '
             'built-in defaults.',
    )
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
# Output file initialisation
# ---------------------------------------------------------------------------

def _write_node_metadata(ds, node_index, node_lon, node_lat, node_depth,
                          point_lon, point_lat, dist_km):
    """Write static node-dimension variables (shared by both output files)."""
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


def _set_global_attrs(ds, title, summary, metadata, extra_attrs, csv_name):
    extra = {'source_csv': csv_name, 'extracted_years': ''}
    extra.update(extra_attrs)
    nc_metadata.set_global_attrs(ds, metadata, title=title, summary=summary,
                                 feature_type='timeSeries', extra=extra)


def init_hourly_output(path, n_nodes, node_index, node_lon, node_lat,
                        node_depth, point_lon, point_lat, dist_km, csv_name,
                        metadata):
    """Create the hourly output NetCDF."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    ds = nc.Dataset(str(out), 'w', format='NETCDF4')
    _set_global_attrs(
        ds,
        title=('STOFS2D-Global hourly water surface elevation at '
               'SurgeMIP official shoreline'),
        summary=('Hourly total water surface elevation extracted from '
                 'ADCIRC fort.63.nc output at the official SurgeMIP '
                 'shoreline points.'),
        metadata=metadata,
        extra_attrs={},
        csv_name=csv_name,
    )
    ds.createDimension('node', n_nodes)
    ds.createDimension('time', None)  # unlimited

    v = ds.createVariable('time', 'f8', ('time',))
    v.standard_name = 'time'
    v.units = nc_metadata.TIME_UNITS
    v.calendar = 'standard'
    v.axis = 'T'

    v = ds.createVariable('zeta', 'f4', ('node', 'time'),
                          zlib=True, complevel=1,
                          chunksizes=(n_nodes, 1),
                          fill_value=CF_FILL_F32)
    v.standard_name = 'sea_surface_height_above_geoid'
    v.long_name = 'Water surface elevation'
    v.units = 'm'
    v.coordinates = 'node_lon node_lat time'
    v.grid_mapping = 'crs'

    _write_node_metadata(ds, node_index, node_lon, node_lat, node_depth,
                         point_lon, point_lat, dist_km)
    nc_metadata.set_geospatial_extent(ds, node_lon, node_lat, node_depth,
                                      positive='down')
    ds.close()
    print(f'Created hourly output: {out}')


def init_monthly_output(path, n_nodes, node_index, node_lon, node_lat,
                         node_depth, point_lon, point_lat, dist_km, csv_name,
                         metadata):
    """Create the monthly maxima output NetCDF."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    ds = nc.Dataset(str(out), 'w', format='NETCDF4')
    _set_global_attrs(
        ds,
        title=('STOFS2D-Global monthly maximum water surface elevation at '
               'SurgeMIP official shoreline'),
        summary=('Monthly maximum total water surface elevation, computed '
                 'from hourly ADCIRC fort.63.nc output at the official '
                 'SurgeMIP shoreline points.'),
        metadata=metadata,
        extra_attrs={},
        csv_name=csv_name,
    )
    ds.createDimension('node', n_nodes)
    ds.createDimension('time', None)  # unlimited, one record per month

    v = ds.createVariable('time', 'f8', ('time',))
    v.standard_name = 'time'
    v.units = nc_metadata.TIME_UNITS
    v.calendar = 'standard'
    v.axis = 'T'
    v.long_name = 'Start of calendar month'

    v = ds.createVariable('monthly_max', 'f4', ('node', 'time'),
                          zlib=True, complevel=1,
                          chunksizes=(n_nodes, 1),
                          fill_value=CF_FILL_F32)
    v.standard_name = 'sea_surface_height_above_geoid'
    v.long_name = 'Monthly maximum water surface elevation'
    v.units = 'm'
    v.cell_methods = 'time: maximum within months'
    v.coordinates = 'node_lon node_lat time'
    v.grid_mapping = 'crs'

    v = ds.createVariable('year', 'i2', ('time',))
    v.long_name = 'Calendar year'

    v = ds.createVariable('month', 'i2', ('time',))
    v.long_name = 'Calendar month (1-12)'

    _write_node_metadata(ds, node_index, node_lon, node_lat, node_depth,
                         point_lon, point_lat, dist_km)
    nc_metadata.set_geospatial_extent(ds, node_lon, node_lat, node_depth,
                                      positive='down')
    ds.close()
    print(f'Created monthly output: {out}')


# ---------------------------------------------------------------------------
# Restart helpers
# ---------------------------------------------------------------------------

def get_extracted_years(path):
    if not Path(path).exists():
        return set()
    ds = nc.Dataset(str(path), 'r')
    years_str = getattr(ds, 'extracted_years', '')
    ds.close()
    if not years_str:
        return set()
    return set(int(y) for y in years_str.split(',') if y.strip())


def mark_year_complete(path, year, existing_years):
    ds = nc.Dataset(str(path), 'a')
    existing_years.add(year)
    ds.extracted_years = ','.join(str(y) for y in sorted(existing_years))
    ds.sync()
    ds.close()


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
# Append to output files
# ---------------------------------------------------------------------------

def append_hourly(path, year_data, times, year, existing_years):
    """
    Append one year of hourly data and update extracted_years.

    year_data : float32 [n_nodes, n_times]
    times : pd.DatetimeIndex
    """
    ds = nc.Dataset(str(path), 'a')
    start = ds.dimensions['time'].size
    n_new = len(times)
    n_nodes = year_data.shape[0]

    # Convert to whatever epoch/calendar this file's `time` variable already
    # declares, so appends stay consistent even if the file was created
    # under a different reference epoch than the pipeline's current default.
    time_vals = nc_metadata.write_times(ds, 'time', times)

    print(f'  Appending hourly {year}: {n_nodes:,} nodes × {n_new} steps '
          f'(offset {start})')

    # Write time axis
    ds.variables['time'][start:start + n_new] = time_vals

    # Write zeta in node chunks to stay within memory
    chunk = 10000
    for i in range(0, n_nodes, chunk):
        j = min(i + chunk, n_nodes)
        ds.variables['zeta'][i:j, start:start + n_new] = year_data[i:j, :]

    nc_metadata.update_time_coverage(ds, times)
    ds.sync()
    existing_years.add(year)
    ds.extracted_years = ','.join(str(y) for y in sorted(existing_years))
    ds.close()


def append_monthly(path, year_data, times, year, existing_years):
    """
    Compute monthly maxima from year_data and append to monthly output.

    year_data : float32 [n_nodes, n_times], CF fill for dry/invalid
    times : pd.DatetimeIndex
    """
    ds = nc.Dataset(str(path), 'a')
    start = ds.dimensions['time'].size

    n_nodes = year_data.shape[0]

    months = sorted(times.month.unique())
    print(f'  Appending monthly {year}: {len(months)} months '
          f'(offset {start})')

    for mi, m in enumerate(months):
        mask = times.month == m
        chunk = year_data[:, mask].copy()  # [n_nodes, n_in_month]

        # Replace CF fill with nan for nanmax, then back to fill
        chunk = chunk.astype(np.float64)
        chunk[chunk > 9e35] = np.nan
        month_max = np.nanmax(chunk, axis=1).astype(np.float32)  # [n_nodes]
        month_max[np.isnan(month_max)] = CF_FILL_F32

        idx = start + mi
        ds.variables['monthly_max'][:, idx] = month_max
        ds.variables['year'][idx] = year
        ds.variables['month'][idx] = m

        # timestamp = first hour of this month
        first_t = times[mask][0]
        ds.variables['time'][idx] = nc_metadata.write_times(ds, 'time', [first_t])[0]

    nc_metadata.update_time_coverage(ds, times)
    ds.sync()
    existing_years.add(year)
    ds.extracted_years = ','.join(str(y) for y in sorted(existing_years))
    ds.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    hourly_path = Path(args.output_hourly)
    monthly_path = Path(args.output_monthly)
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
    metadata = nc_metadata.load_metadata(args.metadata_yaml,
                                         source_attrs=source_attrs)

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

    # Node metadata in CSV row order
    node_lon_csv = node_lon        # already in CSV order (from load_official_points)
    node_lat_csv = node_lat
    node_depth_csv = node_depth
    point_lon_csv = point_lon
    point_lat_csv = point_lat
    dist_km_csv = dist_km

    # --- Determine which years to extract ---
    hourly_years = get_extracted_years(hourly_path)
    monthly_years = get_extracted_years(monthly_path)

    if args.force:
        # Strip all years and start fresh
        if hourly_path.exists():
            hourly_path.unlink()
            print(f'Removed (--force): {hourly_path}')
        if monthly_path.exists():
            monthly_path.unlink()
            print(f'Removed (--force): {monthly_path}')
        hourly_years = set()
        monthly_years = set()
    else:
        done = hourly_years & monthly_years
        if done:
            print(f'Skipping already-extracted years '
                  f'({len(done)}): {", ".join(str(y) for y in sorted(done))}')
        # Years in hourly but not monthly (partial crash): redo both
        partial = hourly_years - monthly_years
        if partial:
            print(f'Partial years (in hourly but not monthly), will re-extract: '
                  f'{", ".join(str(y) for y in sorted(partial))}')
            hourly_years -= partial
            monthly_years -= partial

    todo = [(y, p) for y, p in file_list
            if y not in (hourly_years & monthly_years)]

    if not todo:
        print('All years already extracted. Use --force to redo.')
        return

    # --- Initialise output files if needed ---
    if not hourly_path.exists():
        init_hourly_output(
            hourly_path, n_nodes,
            node_index, node_lon_csv, node_lat_csv, node_depth_csv,
            point_lon_csv, point_lat_csv, dist_km_csv, csv_name, metadata)

    if not monthly_path.exists():
        init_monthly_output(
            monthly_path, n_nodes,
            node_index, node_lon_csv, node_lat_csv, node_depth_csv,
            point_lon_csv, point_lat_csv, dist_km_csv, csv_name, metadata)

    # --- Extract each year ---
    total_t0 = timer.time()
    for i, (year, adcirc_path) in enumerate(todo):
        print(f'\n{"=" * 60}')
        print(f'Year {year}: {adcirc_path}  [{i + 1}/{len(todo)}]')
        print(f'{"=" * 60}')

        times, year_data = extract_year(
            adcirc_path, sorted_nodes, unsort, args.batch_size)

        append_hourly(hourly_path, year_data, times, year, hourly_years)
        append_monthly(monthly_path, year_data, times, year, monthly_years)

        elapsed = timer.time() - total_t0
        print(f'  Year {year} done. Cumulative: {elapsed:.0f}s')

    total_elapsed = timer.time() - total_t0
    print(f'\nAll done. Total: {total_elapsed:.1f}s '
          f'({total_elapsed / 3600:.2f}h)')
    print(f'  Hourly:  {hourly_path}')
    print(f'  Monthly: {monthly_path}')


if __name__ == '__main__':
    main()

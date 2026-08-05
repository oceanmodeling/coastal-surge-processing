"""
Shared utilities for STOFS2D-Global SurgeMIP station extraction and comparison.

Functions for parsing GESLAv4 station files, loading ADCIRC mesh coordinates,
spatial matching of stations to mesh nodes, and computing skill metrics.
"""

import time as timer
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# GESLA file parsing
# ---------------------------------------------------------------------------

def parse_gesla_header(filepath):
    """
    Parse the 41-line GESLA header, returning a dict of metadata.

    Key fields: site_name, site_code, country, latitude, longitude,
    start_date, end_date, datum_info, timezone.
    """
    meta = {}
    with open(filepath, 'r', errors='replace') as f:
        for i, line in enumerate(f):
            if i >= 41:
                break
            line = line.strip()
            if '#' not in line:
                continue
            parts = line.split('#', 1)
            if len(parts) < 2:
                continue
            content = parts[1].strip()

            if content.startswith('SITE NAME'):
                meta['site_name'] = content.split(':', 1)[-1].strip() if ':' in content else content.replace('SITE NAME', '').strip()
            elif content.startswith('SITE CODE'):
                meta['site_code'] = content.split(':', 1)[-1].strip() if ':' in content else content.replace('SITE CODE', '').strip()
            elif content.startswith('COUNTRY'):
                meta['country'] = content.split(':', 1)[-1].strip() if ':' in content else content.replace('COUNTRY', '').strip()
            elif content.startswith('LATITUDE'):
                try:
                    meta['latitude'] = float(content.split()[-1])
                except ValueError:
                    pass
            elif content.startswith('LONGITUDE'):
                try:
                    meta['longitude'] = float(content.split()[-1])
                except ValueError:
                    pass
            elif content.startswith('START DATE'):
                for token in content.split():
                    if '/' in token and len(token) == 10:
                        meta['start_date'] = token
                        break
            elif content.startswith('END DATE'):
                for token in content.split():
                    if '/' in token and len(token) == 10:
                        meta['end_date'] = token
                        break
            elif content.startswith('TIME ZONE'):
                try:
                    meta['timezone'] = float(content.split()[-1])
                except ValueError:
                    meta['timezone'] = 0.0
            elif content.startswith('DATUM'):
                meta['datum'] = content.split(':', 1)[-1].strip() if ':' in content else content.replace('DATUM INFORMATION', '').strip()
    return meta


def _parse_one_station(filepath):
    """Parse a single GESLA file header. Returns a dict or None."""
    try:
        meta = parse_gesla_header(filepath)
    except Exception:
        return None
    if 'latitude' not in meta or 'longitude' not in meta:
        return None
    return {
        'file': str(filepath),
        'filename': Path(filepath).name,
        'site_name': meta.get('site_name', ''),
        'site_code': meta.get('site_code', ''),
        'country': meta.get('country', ''),
        'latitude': meta['latitude'],
        'longitude': meta['longitude'],
        'start_date': meta.get('start_date', ''),
        'end_date': meta.get('end_date', ''),
        'timezone': meta.get('timezone', 0.0),
        'datum': meta.get('datum', ''),
    }


def build_station_catalogue(gesla_dir, t_start=None, t_end=None):
    """
    Scan all GESLA files, parse headers, and return a DataFrame of stations.

    If t_start and t_end are provided, filters to stations whose date range
    overlaps [t_start, t_end].  Uses multiprocessing to parse headers in
    parallel.
    """
    from multiprocessing import Pool, cpu_count

    gesla_path = Path(gesla_dir)
    files = sorted(gesla_path.glob('*'))
    files = [f for f in files if f.is_file() and not f.name.startswith('.')]

    n_workers = min(cpu_count(), 32)
    print(f'  Scanning {len(files)} GESLA files with {n_workers} workers ...')

    with Pool(n_workers) as pool:
        results = pool.map(_parse_one_station, files)

    records = [r for r in results if r is not None]

    cat = pd.DataFrame(records)
    if cat.empty:
        return cat

    if t_start is not None and t_end is not None:
        sim_start = pd.Timestamp(t_start)
        sim_end = pd.Timestamp(t_end)

        def overlaps(row):
            try:
                s = pd.Timestamp(row['start_date'])
                e = pd.Timestamp(row['end_date'])
                return s <= sim_end and e >= sim_start
            except Exception:
                return True
        cat = cat[cat.apply(overlaps, axis=1)].reset_index(drop=True)

    print(f'  {len(cat)} stations found')
    return cat


def read_gesla_data(filepath, t_start=None, t_end=None):
    """
    Read GESLA observation data, returning a DataFrame with a sea_level
    column and a DatetimeIndex.

    Only returns data with use_flag == 1 and valid sea_level values.
    If t_start/t_end are provided, filters to that time range.
    """
    df = pd.read_csv(
        filepath,
        skiprows=41,
        sep=r'\s+',
        header=None,
        names=['date', 'time', 'sea_level', 'qc_flag', 'use_flag'],
        dtype={'date': str, 'time': str, 'sea_level': float,
               'qc_flag': int, 'use_flag': int},
        na_values=['-99.9999'],
    )

    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'],
                                     format='%Y/%m/%d %H:%M:%S',
                                     errors='coerce')
    df = df.dropna(subset=['datetime'])
    df = df.set_index('datetime').sort_index()

    if t_start is not None and t_end is not None:
        df = df.loc[t_start:t_end]

    df = df[df['use_flag'] == 1]
    df = df[df['sea_level'].notna()]

    return df[['sea_level']]


# ---------------------------------------------------------------------------
# Mesh node matching
# ---------------------------------------------------------------------------

def load_mesh_coords(adcirc_path):
    """Load ADCIRC mesh node coordinates (lon, lat) in degrees."""
    import netCDF4 as nc

    ds = nc.Dataset(adcirc_path, 'r')
    x = ds.variables['x'][:]
    y = ds.variables['y'][:]
    ds.close()

    return x, y


def find_nearest_nodes(station_lons, station_lats, mesh_lon, mesh_lat,
                       max_dist_km=50.0):
    """
    For each station, find the nearest mesh node within max_dist_km.

    Returns arrays: node_indices (int, -1 if none), distances_km (float).
    Uses scipy.spatial.cKDTree on a Cartesian (ECEF) projection.
    """
    from scipy.spatial import cKDTree

    R = 6371.0

    def to_xyz(lon, lat):
        lon_r = np.radians(lon)
        lat_r = np.radians(lat)
        return np.column_stack([
            R * np.cos(lat_r) * np.cos(lon_r),
            R * np.cos(lat_r) * np.sin(lon_r),
            R * np.sin(lat_r),
        ])

    print('  Building spatial index of mesh nodes ...')
    t0 = timer.time()
    mesh_xyz = to_xyz(mesh_lon, mesh_lat)
    tree = cKDTree(mesh_xyz)
    print(f'  KDTree built in {timer.time() - t0:.1f}s')

    station_xyz = to_xyz(np.array(station_lons), np.array(station_lats))
    dists, indices = tree.query(station_xyz, k=1)

    dists_km = dists  # already in km since we used R=6371

    node_indices = np.where(dists_km <= max_dist_km, indices, -1)

    n_matched = np.sum(node_indices >= 0)
    print(f'  {n_matched}/{len(station_lons)} stations within {max_dist_km} km '
          f'of a mesh node')

    return node_indices, dists_km


# ---------------------------------------------------------------------------
# ADCIRC time series extraction
# ---------------------------------------------------------------------------

def extract_adcirc_timeseries(adcirc_path, unique_nodes):
    """
    Extract zeta time series for a set of unique node indices from fort.63.nc.

    Returns (times, zeta_dict) where times is a pd.DatetimeIndex and
    zeta_dict maps node_index -> 1D numpy array of zeta values.

    Uses batched time-slice reads to avoid repeated full-file scans.
    """
    import netCDF4 as nc

    ds = nc.Dataset(adcirc_path, 'r')
    cal = getattr(ds.variables['time'], 'calendar', 'standard')
    cftime_dates = nc.num2date(ds.variables['time'][:],
                               ds.variables['time'].units, cal)
    times = pd.to_datetime([d.isoformat() for d in cftime_dates])

    n_times = len(times)
    n_nodes = len(unique_nodes)
    zeta_var = ds.variables['zeta']

    print(f'  Extracting {n_nodes} node time series '
          f'({n_times} timesteps) ...')
    t0 = timer.time()

    sorted_nodes = np.array(sorted(unique_nodes))
    node_min, node_max = int(sorted_nodes[0]), int(sorted_nodes[-1])

    result = np.empty((n_nodes, n_times), dtype=np.float64)

    node_to_col = {n: i for i, n in enumerate(sorted_nodes)}

    span = node_max - node_min + 1
    use_slab = span < 5 * n_nodes

    batch_size = 200
    for b_start in range(0, n_times, batch_size):
        b_end = min(b_start + batch_size, n_times)
        if use_slab:
            slab = zeta_var[b_start:b_end, node_min:node_max + 1]
            local_idx = sorted_nodes - node_min
            result[:, b_start:b_end] = slab[:, local_idx].T
        else:
            for t in range(b_start, b_end):
                row = zeta_var[t, sorted_nodes]
                result[:, t] = row

        elapsed = timer.time() - t0
        pct = b_end / n_times * 100
        print(f'    {b_end}/{n_times} timesteps ({pct:.0f}%) '
              f'[{elapsed:.0f}s]', flush=True)

    ds.close()

    # Mask dry nodes (ADCIRC uses -99999)
    result = np.where(result < -9999, np.nan, result)

    zeta_dict = {}
    for node_idx in unique_nodes:
        zeta_dict[node_idx] = result[node_to_col[node_idx], :]

    elapsed = timer.time() - t0
    print(f'  Extracted in {elapsed:.1f}s')

    return times, zeta_dict


# ---------------------------------------------------------------------------
# Skill metrics
# ---------------------------------------------------------------------------

def compute_metrics(obs, mod):
    """
    Compute comparison metrics between observation and model time series.

    Both inputs are pandas Series with datetime index.
    Returns a dict of metrics, or None if insufficient overlap.
    """
    obs_h = obs.resample('1h').mean().dropna()
    mod_h = mod.resample('1h').mean().dropna()

    common = obs_h.index.intersection(mod_h.index)
    if len(common) < 24:
        return None

    o = obs_h.loc[common].values
    m = mod_h.loc[common].values

    diff = m - o
    bias = np.nanmean(diff)
    rmse = np.sqrt(np.nanmean(diff**2))

    o_anom = o - np.nanmean(o)
    m_anom = m - np.nanmean(m)

    denom = np.sqrt(np.nansum(o_anom**2) * np.nansum(m_anom**2))
    corr = np.nansum(o_anom * m_anom) / denom if denom > 0 else np.nan

    si = rmse / np.nanstd(o) if np.nanstd(o) > 0 else np.nan

    return {
        'n_hours': int(len(common)),
        'bias_m': round(float(bias), 4),
        'rmse_m': round(float(rmse), 4),
        'correlation': round(float(corr), 4),
        'scatter_index': round(float(si), 4),
        'obs_mean_m': round(float(np.nanmean(o)), 4),
        'obs_std_m': round(float(np.nanstd(o)), 4),
        'mod_mean_m': round(float(np.nanmean(m)), 4),
        'mod_std_m': round(float(np.nanstd(m)), 4),
    }

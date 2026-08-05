#!/usr/bin/env python3
"""
Extract 72-hour block maxima of detided surge at SurgeMIP official shoreline
points from a compact hourly water level NetCDF.

This script is Step 2 of the SurgeMIP surge extraction pipeline:

  Step 1 — extract_outputs_to_shoreline_pts.py
      Reads raw ADCIRC fort.63.nc files (one per year), extracts hourly water
      levels at the 35,278 official SurgeMIP shoreline points, and writes a
      compact CF-1.8 NetCDF (~43 GB for 46 years at float32).

  Step 2 — this script
      Reads the compact hourly file produced by Step 1, fits tidal harmonics
      across the full simulation period, subtracts the tidal prediction, and
      computes 72-hour block maxima of the nontidal surge residual.

Algorithm (two-phase, restart-safe):

  Phase 1 — Tidal fit (streams through all years once):
      Builds a vectorized linear tidal design matrix A for each year using
      pytides2 (speeds, nodal factors, equilibrium arguments for all 29
      SurgeMIP constituents).  Accumulates the normal equations

          A^T A   (n_coefs × n_coefs,  tiny — ~60×60)
          A^T Y   (n_coefs × n_nodes,  ~16 MB for 35k nodes)

      across all years, then solves once for the tidal coefficient matrix

          C = (A^T A)^{-1} A^T Y   (n_coefs × n_nodes)

      Checkpointed after each year so a walltime kill can be resumed without
      losing progress.

  Phase 2 — Surge block maxima (streams through all years once more):
      Re-reads each year's compact data, subtracts the tidal prediction

          surge(t) = zeta(t) - A(t) @ C

      and accumulates 72-hour block maxima of the surge signal.  Appended
      per-year for restart safety.

Tidal model:
  Uses the 29-constituent FULL set from the SurgeMIP detide library
  (github.com/oceanmodeling/detide), identical to the set used in the
  SurgeMIP community detiding tools, so results are consistent across
  participants.  Nodal corrections are evaluated every PARTITION_HOURS=240h
  following pytides2 convention.

Usage:
  python extract_surge_block_maxima.py \\
      --compact-file /path/to/cfs_reanalysis_35k_hourly.nc \\
      --output       /path/to/cfs_reanalysis_detided_35k.nc

  # Restart after a killed job — resumes from checkpoint automatically:
  python extract_surge_block_maxima.py \\
      --compact-file /path/to/cfs_reanalysis_35k_hourly.nc \\
      --output       /path/to/cfs_reanalysis_detided_35k.nc

  # Force re-extraction of already-written years:
  python extract_surge_block_maxima.py \\
      --compact-file /path/to/cfs_reanalysis_35k_hourly.nc \\
      --output       /path/to/cfs_reanalysis_detided_35k.nc \\
      --force

Dependencies:
  numpy, netCDF4, pandas,
  pytides2  (pip install git+https://github.com/tomsail/pytides.git)
  detide    (git submodule: third_party/detide)
"""

import argparse
import sys
import time as timer
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd

import nc_metadata

# Import tidal constituent list from the SurgeMIP community detide library.
# This library lives in third_party/detide (git submodule).
sys.path.insert(
    0, str(Path(__file__).resolve().parent / 'third_party' / 'detide'))
from detide.constants import FULL as TIDAL_CONSTITUENTS
from pytides2.astro import astro
from pytides2.tide import Tide


BLOCK_HOURS     = 72     # block length for surge block maxima
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
        '--compact-file', required=True,
        help='Path to the compact hourly NetCDF produced by '
             'extract_outputs_to_shoreline_pts.py',
    )
    p.add_argument(
        '--output', required=True,
        help='Output NetCDF path for 72-hour detided surge block maxima',
    )
    p.add_argument(
        '--force', action='store_true',
        help='Re-extract years already written to the output file',
    )
    p.add_argument(
        '--batch-size', type=int, default=BLOCK_HOURS,
        help=f'Timesteps per slab read (default: {BLOCK_HOURS}, must be a '
             f'multiple of {BLOCK_HOURS}).  Larger values reduce I/O overhead '
             f'at the cost of more memory.',
    )
    p.add_argument(
        '--metadata-yaml',
        help='Path to a YAML file with global NetCDF metadata (institution, '
             'contact, project, license, ...). See metadata_template.yaml '
             'for the editable template. Fields not present fall back to '
             'built-in defaults.',
    )
    args = p.parse_args()
    if args.batch_size % BLOCK_HOURS != 0:
        p.error(f'--batch-size must be a multiple of {BLOCK_HOURS}')
    return args


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

def checkpoint_path(output_path):
    return Path(output_path).with_suffix('.tidal_checkpoint.npz')


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
# Compact file metadata
# ---------------------------------------------------------------------------

def load_compact_file_meta(compact_path):
    """
    Read node metadata and time axis from the compact hourly NetCDF.

    Returns
    -------
    node_index : ndarray int, 0-based ADCIRC node indices
    node_lon, node_lat : ndarray float
    times_by_year : dict {year: pd.DatetimeIndex}
    """
    ds = nc.Dataset(str(compact_path), 'r')
    node_index = np.array(ds.variables['node_index'][:], dtype=np.int64) - 1
    node_lon   = np.array(ds.variables['node_lon'][:], dtype=np.float64)
    node_lat   = np.array(ds.variables['node_lat'][:], dtype=np.float64)

    epoch      = pd.Timestamp('1970-01-01')
    time_hours = np.array(ds.variables['time'][:], dtype=np.float64)
    times      = epoch + pd.to_timedelta(time_hours, unit='h')
    ds.close()

    years = pd.DatetimeIndex(times).year
    unique_years = sorted(set(years))
    times_by_year = {yr: pd.DatetimeIndex(times[years == yr])
                     for yr in unique_years}

    print(f'Compact file: {compact_path}')
    print(f'  {len(node_index):,} nodes, {len(times):,} timesteps, '
          f'{len(unique_years)} years ({unique_years[0]}–{unique_years[-1]})')

    return node_index, node_lon, node_lat, times_by_year


# ---------------------------------------------------------------------------
# Phase 1: accumulate normal equations and solve for tidal coefficients
# ---------------------------------------------------------------------------

def run_phase1(compact_path, times_by_year, local_positions,
               output_path, batch_size):
    """
    Stream through the compact file once to fit tidal harmonics.

    Returns (C, global_t0, constituents, total_hours) where
    C is shape (n_coefs, n_nodes).
    """
    ckpt_path   = checkpoint_path(output_path)
    all_years   = sorted(times_by_year.keys())
    first_times = times_by_year[all_years[0]]
    last_times  = times_by_year[all_years[-1]]

    global_t0     = first_times[0].to_pydatetime()
    global_t0_iso = global_t0.isoformat()
    total_hours   = (last_times[-1] - first_times[0]).total_seconds() / 3600.0

    print(f'Global reference time: {global_t0_iso}')
    print(f'Total time span: {total_hours:.0f}h ({total_hours/8760:.1f} yr)')

    constituents     = filter_constituents(TIDAL_CONSTITUENTS, total_hours,
                                           global_t0)
    n_coefs          = 2 * len(constituents) + 2
    n_nodes          = len(local_positions)
    constituent_names = [c.name for c in constituents]
    print(f'{len(constituents)} tidal constituents: '
          f'{", ".join(constituent_names)}')

    ATA = np.zeros((n_coefs, n_coefs), dtype=np.float64)
    ATY = np.zeros((n_coefs, n_nodes), dtype=np.float64)
    accumulated_years = set()

    # Resume from checkpoint if available and compatible
    if ckpt_path.exists():
        ckpt = load_checkpoint(ckpt_path)
        if (ckpt['global_t0_iso'] == global_t0_iso and
                list(ckpt['constituent_names']) == constituent_names):
            ATA, ATY = ckpt['ATA'], ckpt['ATY']
            accumulated_years = ckpt['accumulated_years']
            print(f'Resumed from checkpoint: '
                  f'{len(accumulated_years)} years already accumulated')
        else:
            print('Checkpoint incompatible (different t0 or constituents), '
                  'starting fresh')

    phase1_t0 = timer.time()
    for i, year in enumerate(all_years):
        if year in accumulated_years:
            print(f'  Year {year}: already accumulated, skipping')
            continue

        print(f'\n--- Phase 1: Year {year} [{i+1}/{len(all_years)}] ---')

        year_times = times_by_year[year]
        hours      = (year_times - pd.Timestamp(global_t0)).total_seconds().values \
                     / 3600.0
        A          = build_tidal_design_matrix(hours, global_t0,
                                               constituents, total_hours)
        t_offset   = sum(len(times_by_year[yr])
                         for yr in all_years if yr < year)

        _accumulate_year(compact_path, local_positions, A, ATA, ATY,
                         batch_size, t_offset)

        accumulated_years.add(year)
        save_checkpoint(ckpt_path, ATA, ATY, global_t0_iso,
                        accumulated_years, constituent_names)

    print(f'\nPhase 1 complete in {timer.time()-phase1_t0:.1f}s')
    print('Solving tidal coefficients ...')
    C = np.linalg.solve(ATA, ATY)
    print(f'  C shape: {C.shape}')
    return C, global_t0, constituents, total_hours


def _accumulate_year(compact_path, local_positions, A, ATA, ATY,
                     batch_size, t_offset):
    """Accumulate one year's normal equations from the compact file."""
    ds       = nc.Dataset(str(compact_path), 'r')
    zeta_var = ds.variables['zeta']   # (node, time)
    n_times  = A.shape[0]

    ATA += A.T @ A

    t0_wall      = timer.time()
    report_every = max(batch_size, BLOCK_HOURS * 10)

    for t_s in range(0, n_times, batch_size):
        t_e  = min(t_s + batch_size, n_times)
        slab = np.array(
            zeta_var[local_positions, t_offset + t_s:t_offset + t_e],
            dtype=np.float64)                    # [n_nodes, B]
        vals = slab.T                            # [B, n_nodes]
        del slab
        vals[vals > CF_FILL_F32 / 2] = 0.0      # zero-out dry sentinels

        ATY += A[t_s:t_e, :].T @ vals           # DGEMM

        if t_e % report_every < batch_size or t_e == n_times:
            elapsed = timer.time() - t0_wall
            print(f'    [{t_e/n_times*100:.0f}%] {t_e}/{n_times} '
                  f'[{elapsed:.0f}s]', flush=True)

    ds.close()
    print(f'  Accumulated in {timer.time()-t0_wall:.1f}s')


# ---------------------------------------------------------------------------
# Phase 2: detided 72-hour block maxima
# ---------------------------------------------------------------------------

def run_phase2(compact_path, times_by_year, local_positions, C,
               global_t0, constituents, total_hours, output_path,
               batch_size, force, metadata):
    """
    Subtract tidal prediction and compute 72-hour block maxima for each year.
    """
    all_years      = sorted(times_by_year.keys())
    existing_years = _get_existing_years(output_path)

    if existing_years:
        print(f'Output has years: {sorted(existing_years)}')

    phase2_years = list(all_years)
    if not force:
        skip = [y for y in phase2_years if y in existing_years]
        if skip:
            print(f'Skipping already-extracted years: {skip} '
                  f'(use --force to re-extract)')
        phase2_years = [y for y in phase2_years if y not in existing_years]
    else:
        force_years = set(phase2_years) & existing_years
        if force_years and Path(output_path).exists():
            _strip_years(output_path, force_years)

    if not phase2_years:
        print('Nothing to extract in Phase 2.')
        return

    total_t0 = timer.time()
    for i, year in enumerate(phase2_years):
        print(f'\n{"="*60}')
        print(f'Phase 2: Year {year} [{i+1}/{len(phase2_years)}]')
        print(f'{"="*60}')

        year_times = times_by_year[year]
        n_times    = len(year_times)
        hours      = (year_times - pd.Timestamp(global_t0)).total_seconds().values \
                     / 3600.0
        A          = build_tidal_design_matrix(hours, global_t0,
                                               constituents, total_hours)
        t_offset   = sum(len(times_by_year[yr])
                         for yr in all_years if yr < year)

        bm, bt = _extract_block_maxima(
            compact_path, local_positions, A, C, year_times,
            t_offset, batch_size)

        if not Path(output_path).exists():
            # node_index and coords come from the first local_positions call
            ds0 = nc.Dataset(str(compact_path), 'r')
            node_index = np.array(ds0.variables['node_index'][:],
                                  dtype=np.int64) - 1
            node_lon   = np.array(ds0.variables['node_lon'][:])
            node_lat   = np.array(ds0.variables['node_lat'][:])
            ds0.close()
            _init_output(output_path, node_index[local_positions],
                         node_lon[local_positions], node_lat[local_positions],
                         constituents, metadata)

        _append_year(output_path, bm, bt, year)

    print(f'\nPhase 2 total: {timer.time()-total_t0:.1f}s')


def _extract_block_maxima(compact_path, local_positions, A, C,
                          year_times, t_offset, batch_size):
    """Extract surge block maxima for one year from the compact file."""
    ds       = nc.Dataset(str(compact_path), 'r')
    zeta_var = ds.variables['zeta']
    n_times  = len(year_times)
    n_nodes  = len(local_positions)
    n_blocks = (n_times + BLOCK_HOURS - 1) // BLOCK_HOURS

    block_maxima     = np.full((n_nodes, n_blocks), np.nan, dtype=np.float32)
    block_center_times = []

    print(f'  {n_times:,} timesteps → {n_blocks} blocks '
          f'(batch={batch_size})')
    t0_wall = timer.time()

    for b_s in range(0, n_blocks, batch_size // BLOCK_HOURS):
        b_e = min(b_s + batch_size // BLOCK_HOURS, n_blocks)
        t_s = b_s * BLOCK_HOURS
        t_e = min(b_e * BLOCK_HOURS, n_times)
        B   = t_e - t_s

        slab = np.array(
            zeta_var[local_positions, t_offset + t_s:t_offset + t_e],
            dtype=np.float64)                    # [n_nodes, B]
        vals = slab.T                            # [B, n_nodes]
        del slab
        valid = vals < CF_FILL_F32 / 2          # [B, n_nodes]

        tide_batch = A[t_s:t_e, :] @ C          # DGEMM [B,n_coef]@[n_coef,N]
        surge      = np.where(valid, vals - tide_batch, -np.inf)
        del tide_batch

        for bi in range(b_s, b_e):
            r_s         = (bi - b_s) * BLOCK_HOURS
            r_e         = min(r_s + BLOCK_HOURS, B)
            block_surge = surge[r_s:r_e, :]
            block_valid = valid[r_s:r_e, :]
            block_max   = np.max(block_surge, axis=0)
            block_max[~np.any(block_valid, axis=0)] = np.nan
            block_maxima[:, bi] = block_max.astype(np.float32)

            mid_t = bi * BLOCK_HOURS + (min(BLOCK_HOURS, n_times - bi*BLOCK_HOURS) - 1) // 2
            block_center_times.append(year_times[mid_t])

        elapsed = timer.time() - t0_wall
        print(f'    [{t_e/n_times*100:.0f}%] t={t_e}/{n_times}, '
              f'blocks {b_s+1}–{b_e}/{n_blocks} [{elapsed:.0f}s]', flush=True)

    ds.close()
    print(f'  Extracted {n_blocks} blocks in {timer.time()-t0_wall:.1f}s')
    return block_maxima, pd.DatetimeIndex(block_center_times)


# ---------------------------------------------------------------------------
# Output I/O (restart-safe: appended per year)
# ---------------------------------------------------------------------------

def _get_existing_years(output_path):
    if not Path(output_path).exists():
        return set()
    ds        = nc.Dataset(str(output_path), 'r')
    years_str = getattr(ds, 'extracted_years', '')
    ds.close()
    return set(int(y) for y in years_str.split(',') if y.strip()) \
        if years_str else set()


def _init_output(output_path, node_index, node_lon, node_lat, constituents,
                 metadata):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    n_nodes = len(node_index)

    ds = nc.Dataset(str(output_path), 'w', format='NETCDF4')
    ds.createDimension('node',  n_nodes)
    ds.createDimension('block', None)   # unlimited

    extra = {
        'block_hours': BLOCK_HOURS,
        'extracted_years': '',
        'detiding_method': (
            'Vectorized linear least-squares harmonic analysis across all '
            'years. Tidal coefficients: C = (A^T A)^{-1} A^T Y. Nodal '
            f'corrections every {PARTITION_HOURS}h via pytides2.'),
        'detiding_constituents': ', '.join(c.name for c in constituents),
    }
    nc_metadata.set_global_attrs(
        ds, metadata,
        title='SurgeMIP 72-hour detided surge block maxima',
        summary=('72-hour block maxima of nontidal surge, computed by '
                 'subtracting a least-squares tidal harmonic fit from '
                 'hourly ADCIRC water levels at the official SurgeMIP '
                 'shoreline points.'),
        feature_type='timeSeries',
        extra=extra,
    )
    nc_metadata.set_geospatial_extent(ds, node_lon, node_lat)

    for name, data, units in [
        ('node_index', node_index.astype(np.int32), None),
        ('node_lon',   node_lon.astype(np.float64), 'degrees_east'),
        ('node_lat',   node_lat.astype(np.float64), 'degrees_north'),
    ]:
        dtype = 'i4' if name == 'node_index' else 'f8'
        v = ds.createVariable(name, dtype, ('node',), zlib=True, complevel=1)
        if units:
            v.units = units
        v[:] = data

    v = ds.createVariable('block_max', 'f4', ('node', 'block'),
                          zlib=True, complevel=1,
                          chunksizes=(min(n_nodes, 10000), 122))
    v.long_name = f'Maximum detided surge in {BLOCK_HOURS}-hour block'
    v.units     = 'm'

    v = ds.createVariable('block_time', 'f8', ('block',))
    v.units    = 'hours since 1970-01-01 00:00:00'
    v.calendar = 'standard'
    v.long_name = f'Center time of {BLOCK_HOURS}-hour block'

    v = ds.createVariable('year', 'i2', ('block',))
    v.long_name = 'Year of simulation'

    ds.close()
    print(f'Created {output_path}')


def _append_year(output_path, block_maxima, block_times, year):
    ds        = nc.Dataset(str(output_path), 'a')
    years_str = getattr(ds, 'extracted_years', '')
    committed = set(int(y) for y in years_str.split(',') if y.strip()) \
        if years_str else set()

    # Find write offset — append after last committed block
    n_total = ds.dimensions['block'].size
    if committed and n_total > 0:
        year_var      = np.array(ds.variables['year'][:])
        committed_idx = np.where(np.isin(year_var, list(committed)))[0]
        start         = int(committed_idx.max()) + 1 if len(committed_idx) else 0
    else:
        start = 0

    n_new    = block_maxima.shape[1]
    n_nodes  = block_maxima.shape[0]
    print(f'\nAppending year {year}: {n_nodes:,} nodes × {n_new} blocks '
          f'(offset {start})')

    chunk = 100_000
    for i in range(0, n_nodes, chunk):
        j = min(i + chunk, n_nodes)
        ds.variables['block_max'][i:j, start:start + n_new] = \
            block_maxima[i:j, :]

    epoch      = pd.Timestamp('1970-01-01')
    time_units = ds.variables['block_time'].units
    time_vals  = nc.date2num(block_times.to_pydatetime(), time_units)
    ds.variables['block_time'][start:start + n_new] = time_vals
    ds.variables['year'][start:start + n_new] = year

    nc_metadata.update_time_coverage(ds, block_times)
    ds.sync()
    committed.add(year)
    ds.extracted_years = ','.join(str(y) for y in sorted(committed))
    ds.close()
    print(f'  Done. {start + n_new} total blocks.')


def _strip_years(output_path, years_to_strip):
    """Rewrite output file without blocks from given years (for --force)."""
    out_path = Path(output_path)
    ds_old   = nc.Dataset(str(out_path), 'r')
    old_years = np.array(ds_old.variables['year'][:])
    keep_mask = ~np.isin(old_years, list(years_to_strip))
    keep_idx  = np.where(keep_mask)[0]
    n_keep    = len(keep_idx)
    n_nodes   = ds_old.dimensions['node'].size

    years_attr = getattr(ds_old, 'extracted_years', '')
    remaining  = (set(int(y) for y in years_attr.split(',') if y.strip())
                  - years_to_strip) if years_attr else set()

    tmp_path = out_path.with_suffix('.nc.tmp')
    ds_new   = nc.Dataset(str(tmp_path), 'w', format='NETCDF4')
    ds_new.createDimension('node',  n_nodes)
    ds_new.createDimension('block', None)

    for attr in ds_old.ncattrs():
        setattr(ds_new, attr, getattr(ds_old, attr))
    ds_new.extracted_years = ','.join(str(y) for y in sorted(remaining))

    for name in ('node_index', 'node_lon', 'node_lat'):
        ov = ds_old.variables[name]
        v  = ds_new.createVariable(name, ov.dtype, ov.dimensions,
                                   zlib=True, complevel=1)
        v[:] = ov[:]

    v = ds_new.createVariable('block_max', 'f4', ('node', 'block'),
                              zlib=True, complevel=1,
                              chunksizes=(min(n_nodes, 10000), 122))
    if n_keep:
        chunk = 100_000
        for i in range(0, n_nodes, chunk):
            j = min(i + chunk, n_nodes)
            v[i:j, :n_keep] = ds_old.variables['block_max'][i:j, keep_idx]

    ov = ds_old.variables['block_time']
    bv = ds_new.createVariable('block_time', 'f8', ('block',))
    bv.units = ov.units;  bv.calendar = getattr(ov, 'calendar', 'standard')
    bv.long_name = ov.long_name
    if n_keep:
        bv[:n_keep] = ov[keep_idx]

    yv = ds_new.createVariable('year', 'i2', ('block',))
    yv.long_name = 'Year of simulation'
    if n_keep:
        yv[:n_keep] = old_years[keep_idx]

    ds_old.close();  ds_new.close()
    tmp_path.rename(out_path)
    print(f'Stripped {years_to_strip} from output.')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    compact_path = Path(args.compact_file)
    output_path  = Path(args.output)
    batch_size   = args.batch_size

    if not compact_path.exists():
        print(f'Compact file not found: {compact_path}')
        sys.exit(1)

    # Inherit any matching global attrs already present on the compact
    # hourly file (set by extract_outputs_to_shoreline_pts.py) when no
    # --metadata-yaml is given, or to fill gaps left by one.
    source_attrs = nc_metadata.read_known_attrs(compact_path)
    metadata = nc_metadata.load_metadata(args.metadata_yaml,
                                         source_attrs=source_attrs)

    node_index, node_lon, node_lat, times_by_year = \
        load_compact_file_meta(compact_path)

    # local_positions: row indices into the compact file's node dimension
    # (0..n_nodes-1, since the compact file already holds only the 35k nodes)
    local_positions = np.arange(len(node_index), dtype=np.int64)

    print(f'\n{"="*60}')
    print('PHASE 1: Tidal harmonic analysis')
    print(f'{"="*60}')
    C, global_t0, constituents, total_hours = run_phase1(
        compact_path, times_by_year, local_positions, output_path, batch_size)

    print(f'\n{"="*60}')
    print('PHASE 2: Detided surge block maxima')
    print(f'{"="*60}')
    run_phase2(compact_path, times_by_year, local_positions, C,
               global_t0, constituents, total_hours, output_path,
               batch_size, args.force, metadata)

    # Remove Phase 1 checkpoint on successful completion
    ckpt = checkpoint_path(output_path)
    if ckpt.exists():
        ckpt.unlink()
        print(f'Removed checkpoint: {ckpt}')

    print('\nDone.')


if __name__ == '__main__':
    main()

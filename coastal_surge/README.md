# SurgeMIP Surge Extraction Pipeline

Two-step pipeline for extracting 72-hour detided surge block maxima at the
35,278 official SurgeMIP shoreline points from ADCIRC `fort.63.nc` output.

---

## Contents

| File | Description |
|------|-------------|
| `extract_outputs_to_shoreline_pts.py` | Step 1: extract hourly water levels from fort.63.nc |
| `extract_surge_block_maxima.py` | Step 2: fit tides and compute surge block maxima |
| `coastal_points_gsshs_low_20km_35k-pts.csv` | Official SurgeMIP 35k shoreline point locations |
| `nc_metadata.py` | Shared helper for CF/ACDD-style global NetCDF metadata |
| `metadata_template.yaml` | Editable template for institution/contact/license/etc. metadata |

---

## Dependencies

```
numpy
netCDF4
pandas
scipy
pyyaml
pytides2    # pip install git+https://github.com/WPringle/pytides.git@add-tidal-constituents
detide      # git submodule: detide_extended_constituents
```

The `detide` library provides the tidal constituent sets used for harmonic
analysis. It's tracked as the `detide_extended_constituents` git submodule
at the repo root (sibling of this `coastal_surge/` directory), pointing at
[github.com/WPringle/detide](https://github.com/WPringle/detide), branch
`adding-extended-constituents`. Initialize it with:

```bash
git submodule update --init --recursive
```

`extract_surge_block_maxima.py` adds `<repo_root>/detide_extended_constituents`
to `sys.path` and imports `detide.constants.EXTENDED` from it — the
67-constituent extended set (see **Tidal model** below). If you relocate the
submodule, edit the `sys.path.insert` line in that script to match.

Install the `pytides2` fork above to match what `detide_extended_constituents`'s
own `pyproject.toml` pins, so your environment stays consistent with the
submodule.

---

## Step 1 — Extract hourly water levels

Reads all `fort.63.nc` files in a campaign directory tree (one subdirectory
per year, e.g. `output/CFS-reanalysis/1979/fort.63.nc`), matches the 35k
shoreline points to the nearest ADCIRC mesh node via ECEF KDTree, and writes
a compact CF-1.8 NetCDF.

```bash
python extract_outputs_to_shoreline_pts.py \
    --points-csv coastal_points_gsshs_low_20km_35k-pts.csv \
    --adcirc-dir /path/to/output/CFS-reanalysis/ \
    --output-hourly  /path/to/cfs_reanalysis_35k_hourly.nc \
    --output-monthly /path/to/cfs_reanalysis_35k_monthly_max.nc
```

**Outputs:**
- `*_35k_hourly.nc` — hourly zeta at 35k nodes, dimensions `(node, time)`,
  chunked `(n_nodes, 1)` for efficient downstream column reads. ~43 GB for
  46 years.
- `*_35k_monthly_max.nc` — monthly maximum water levels. ~60 MB for 46 years.

**Memory:** The full year buffer is held in RAM (~1.2 GB at float32 for 35k
nodes × 8784 hourly timesteps). Batch reads of `--batch-size` rows are used
to control I/O chunk sizes.

**Restart safety:** Completed years are tracked via a `completed_years.json`
sidecar file. Restarting the script will skip already-extracted years.

---

## Step 2 — Fit tides and extract surge block maxima

Reads the compact hourly file from Step 1, fits tidal harmonics across the
full simulation period using a vectorized linear least-squares approach, and
computes 72-hour block maxima of the nontidal surge residual.

```bash
python extract_surge_block_maxima.py \
    --compact-file /path/to/cfs_reanalysis_35k_hourly.nc \
    --output       /path/to/cfs_reanalysis_detided_35k.nc
```

**Output:** `*_detided_35k.nc` — dimensions `(node, block)`, where each
block is the maximum detided surge over a non-overlapping 72-hour window.
Includes `block_time` (center time of each block) and `year` variables.
~700 MB for 46 years.

**Tidal model:** Vectorized linear least-squares harmonic analysis using
the 67-constituent EXTENDED set from the `detide_extended_constituents`
submodule, which allows for better analysis in some shallow regions and
high latitude regions where seasonal influences are important. Constituents
with fewer than 2 complete cycles over the full simulation period are
excluded. Speeds, nodal factors, and equilibrium arguments are computed via
pytides2, with nodal corrections updated every 240 hours following the
pytides2 convention. This approach scales to 35k nodes simultaneously
rather than fitting one node at a time.

**Algorithm:**

*Phase 1 — Tidal fit:*
For each year, builds the tidal design matrix A and accumulates the normal
equations (AᵀA and AᵀY) across all years. After all years are processed,
solves C = (AᵀA)⁻¹AᵀY once to obtain tidal coefficients at all 35k nodes
simultaneously. AᵀA is ~136×136 (tiny); AᵀY is ~136×35k (~36 MB). A Phase 1
checkpoint is saved after each year so a killed job can resume.

*Phase 2 — Surge block maxima:*
Re-reads each year's compact data, subtracts the tidal prediction
`tide(t) = A(t) @ C`, and computes the 72-hour block maximum of the
surge residual. Results are appended to the output file year by year, so
partial runs can be safely resumed.

**Memory:** Phase 1 uses ~36 MB for AᵀY (negligible). Phase 2 holds one
year of block maxima in RAM (~35k × 122 blocks × 4 bytes ≈ 17 MB).

**Restart safety:** Phase 1 checkpoints to `*.tidal_checkpoint.npz` after
each year. Phase 2 appends per year with the completed year list stored in
the output file's `extracted_years` global attribute. Both phases skip
already-completed years on restart.

---

## NetCDF metadata

Both scripts accept an optional `--metadata-yaml path/to/file.yaml` flag
that fills in CF/ACDD-style global attributes (institution, contact,
project, license, keywords, ...) on every output file. Copy
`metadata_template.yaml`, fill in your details, and pass it in:

```bash
python extract_outputs_to_shoreline_pts.py ... --metadata-yaml my_metadata.yaml
python extract_surge_block_maxima.py       ... --metadata-yaml my_metadata.yaml
```

If `--metadata-yaml` is omitted entirely, both scripts still check the
input file they're reading for matching global attributes and copy those
over automatically:
- Step 1 checks the first `fort.63.nc` it reads.
- Step 2 checks the compact hourly file from Step 1 — so metadata set (or
  inherited) in Step 1 propagates to Step 2's output for free.

Precedence is: built-in defaults → attributes copied from the input file →
fields set in `--metadata-yaml`. A YAML file always wins if you want to
override what an input file already has. (The `id` field is never
auto-copied, since it's meant to uniquely identify each output product.)

Any field still left blank falls back to a built-in default (see
`nc_metadata.py`). Fields that describe the data itself rather than its
provenance — `geospatial_lat/lon/vertical_min/max`, `time_coverage_start/
end`, `date_created`, `title`, `summary` — are computed automatically from
the extracted data and are not part of the YAML template.

---

## Output format

Both output files follow CF-1.8 conventions. Key variables:

**`*_35k_hourly.nc`** (Step 1):
| Variable | Dimensions | Description |
|----------|-----------|-------------|
| `zeta` | (node, time) | Hourly water surface elevation (m) |
| `time` | (time,) | Hours since 1900-01-01 |
| `node_index` | (node,) | 0-based ADCIRC mesh node index |
| `node_lon/lat` | (node,) | Matched mesh node coordinates |
| `point_lon/lat` | (node,) | Original CSV point coordinates |
| `dist_km` | (node,) | Distance from CSV point to matched node (km) |

**`*_detided_35k.nc`** (Step 2):
| Variable | Dimensions | Description |
|----------|-----------|-------------|
| `block_max` | (node, block) | 72-hour maximum detided surge (m) |
| `block_time` | (block,) | Block center time (hours since 1900-01-01) |
| `year` | (block,) | Year of simulation |
| `node_index` | (node,) | 0-based ADCIRC mesh node index |
| `node_lon/lat` | (node,) | Mesh node coordinates |

Global attributes on `*_detided_35k.nc` document the detiding method and
constituent list used.

---

## Typical runtimes (Crux/ALCF, Lustre filesystem)

| Step | Nodes | Years | Runtime |
|------|-------|-------|---------|
| Step 1 (extraction) | 35k | 46 | ~12h |
| Step 2, Phase 1 (tidal fit) | 35k | 46 | ~3h |
| Step 2, Phase 2 (block maxima) | 35k | 46 | ~8 min |

Step 1 is I/O-bound (reads ~25 TB of raw fort.63.nc). Step 2 reads the
43 GB compact file twice and is much faster.

---

## Contact

Coleman P. Blakely — cblakely@anl.gov  
Argonne National Laboratory

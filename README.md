# coastal-surge-processing
Consistent and transparent processing of modeled coastal water levels and surge time series

--- 

Created as part of the _Surge Model Intercomparison Project_ 
([SurgeMIP](https://www.sciencedirect.com/science/article/pii/S2212094724000501))
## Quick start

```bash
# install pixi on your system: https://pixi.prefix.dev/latest/installation/
git clone https://github.com/oceanmodeling/coastal-surge-processing.git 
git submodule update --init
pixi install

# Naming-convention fields (group_name/climate_forcing/scenario/location) and
# other provenance can be set once in a YAML file (see
# coastal_surge/metadata_template.yaml) and/or overridden per-run with
# --group-name/--climate-forcing/--scenario/--location.

# Step 1: hourly total water level (twl), one file per year
pixi run python coastal_surge/extract_outputs_to_shoreline_pts.py \
    --points-csv ./data/coastal_points_gsshs_low_20km_35k-pts.csv \
    --adcirc-dir /path/to/output/CFS-reanalysis/ \
    --output-dir /path/to/twl_hourly/ \
    --metadata-yaml coastal_surge/metadata_template.yaml \
    --group-name Argonne --climate-forcing CFSv2 --scenario Reanalysis \
    --location GESLA

# Step 2 (optional): detide via harmonic analysis -> hourly storm surge
# height (ssgh), one file per year. Skip this step if only twl is needed.
pixi run python coastal_surge/detide_surge.py \
    --hourly-dir /path/to/twl_hourly/ \
    --output-dir /path/to/ssgh_hourly/

# Step 3: full-period monthly maxima, for either twl or ssgh
pixi run python coastal_surge/compute_monthly_max.py \
    --hourly-dir /path/to/ssgh_hourly/ \
    --variable   StormSurge \
    --output-dir /path/to/monthly_max/
```

Output files follow a CMIP6-style naming convention (see
https://help.ceda.ac.uk/article/4801-cmip6-data), variable first and time
range last:
`Variable_TimeStep_GroupName_ClimateForcing_Scenario_Location_TimeRange.nc`,
e.g. `twl_Hourly_Argonne_CFSv2_Reanalysis_GESLA_200001-200012.nc` or
`ssgh_MonthlyMax_Argonne_CFSv2_Reanalysis_GESLA_197901-202512.nc`.

| Variable | Short name | CF-style `standard_name` | Meaning |
|----------|-----------|---------------------------|---------|
| WaterLevel | `twl` | `total_water_level` | Astronomical + meteorological driven water level (storm tide); see file metadata for model-specific contributions. |
| StormSurge | `ssgh` | `storm_surge_height` | Non-tidal residual of `twl`. Preference is to subtract an astronomical-tide-only model run where available; `detide_surge.py` implements the harmonic-analysis fallback (EXTENDED constituent set minus Sa/Ssa — see [Detiding](#detiding) below). |

### Detiding

Sa and Ssa are excluded from the harmonic tidal fit used to compute
StormSurge: both are mostly non-astronomical (seasonal/meteorological) in
origin, so including them would strip real seasonal surge signal rather than
tide. The remaining 65-constituent EXTENDED set (Pengcheng Wang's list,
Rayleigh criterion 0.8) is used as-is. See `detide_surge.py`'s docstring and
the `detiding_method`/`detiding_constituents` global attributes on its
output files for details.

## Docs

Methodology, API reference, and open questions: 
**[oceanmodeling.github.io/detide](https://oceanmodeling.github.io/detide/)**

## Contributing

Issues and PRs welcome - especially discussion on constituent sets, metadata fields, and validation approaches.


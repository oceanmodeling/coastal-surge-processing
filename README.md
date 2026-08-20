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

pixi run python coastal_surge/extract_outputs_to_shoreline_pts.py \
    --points-csv ./data/coastal_points_gsshs_low_20km_35k-pts.csv \
    --adcirc-dir /path/to/output/CFS-reanalysis/ \
    --output-hourly  /path/to/cfs_reanalysis_35k_hourly.nc \
    --output-monthly /path/to/cfs_reanalysis_35k_monthly_max.nc \
    --metadata-yaml  coastal_surge/metadata_template.yaml

pixi run python coastal_surge/extract_surge_block_maxima.py \
    --compact-file /path/to/cfs_reanalysis_35k_hourly.nc \
    --output       /path/to/cfs_reanalysis_detided_35k.nc
```

## Docs

Methodology, API reference, and open questions: 
**[oceanmodeling.github.io/detide](https://oceanmodeling.github.io/detide/)**

## Contributing

Issues and PRs welcome - especially discussion on constituent sets, metadata fields, and validation approaches.


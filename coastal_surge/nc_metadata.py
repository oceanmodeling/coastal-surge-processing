#!/usr/bin/env python3
"""
Shared CF/ACDD-style global metadata for the SurgeMIP pipeline's NetCDF
outputs.

User-facing fields (institution, contact, creator/researcher identity,
project, license, forcing, ...) come from a YAML file — see
metadata_template.yaml for the editable template and a description of each
field. Fields the user leaves out fall back to whatever
matching global attributes already exist on the input file being read (e.g.
an upstream fort.63.nc, or a compact file from an earlier pipeline step),
then to the DEFAULTS below. Fields that are computed from the data itself
(geospatial extent, time coverage, date_created) are set separately by
set_geospatial_extent() and update_time_coverage(), since they can't come
from a static YAML file or be copied from an input file.
"""

import re
from datetime import datetime, timezone
from pathlib import Path

import netCDF4 as nc
import pandas as pd
import yaml

# Reference epoch for all "hours since ..." time variables in the pipeline.
# 1900-01-01 (rather than 1970-01-01) so campaigns that run further back in
# time don't produce negative time values.
EPOCH = pd.Timestamp('1900-01-01')
TIME_UNITS = 'hours since 1900-01-01 00:00:00'

# Fallback values used for any field not present in the user's YAML file.
DEFAULTS = {
    'id': 'SurgeMIP_shoreline_waterlevels',
    'project': 'SurgeMIP',
    'institution': 'Argonne National Laboratory',
    'institution_id': '',
    'contact': '',
    'creator_name': '',
    'creator_id': '',
    'creator_email': '',
    'researcher_name': '',
    'researcher_id': '',
    'researcher_email': '',
    'researcher_affiliation': '',
    'source': 'ADCIRC STOFS2D-Global, CFS reanalysis atmospheric forcing',
    'forcing': '',
    'crs': 'WGS84',
    'license': '',
    'acknowledgment': '',
    'references': '',
    'comment': '',
    'sea_name': 'global',
    'keywords': 'storm surge; coastal flooding; water level; ADCIRC; SurgeMIP',
    'standard_name_vocabulary': 'CF Standard Name Table',
    # Short, filename-safe tokens used to build the SurgeMIP output filename
    # convention (see NAMING_FIELDS/build_filename below). Distinct from the
    # free-text institution/forcing/source provenance fields above.
    'group_name': '',
    'climate_forcing': '',
    'scenario': '',
    'location': '',
}

# Canonical variable definitions for the SurgeMIP naming/CF-attribute
# convention. 'name' is both the netCDF variable name and the filename
# token; 'standard_name' is written as the variable's standard_name
# attribute even though these two terms are not (yet) part of the official
# CF Standard Name Table.
VARIABLES = {
    'WaterLevel': {
        'name': 'twl',
        'standard_name': 'total_water_level',
        'long_name': 'Total Water Level',
    },
    'StormSurge': {
        'name': 'ssgh',
        'standard_name': 'storm_surge_height',
        'long_name': 'Storm Surge Height',
    },
}

# Fields required to build the CMIP6-style
# Variable_TimeStep_GroupName_ClimateForcing_Scenario_Location_TimeRange
# filename convention (see build_filename), in the order they appear
# between the TimeStep and TimeRange tokens.
NAMING_FIELDS = ['group_name', 'climate_forcing', 'scenario', 'location']

# Order in which global attributes are written, matching the style of the
# GTSMv3 reanalysis reference file this template is based on.
_ATTR_ORDER = [
    'id', 'project', 'acknowledgment', 'contact',
    'creator_name', 'creator_id', 'creator_email',
    'researcher_name', 'researcher_id', 'researcher_email',
    'researcher_affiliation',
    'license', 'institution', 'institution_id', 'sea_name', 'source',
    'forcing', 'crs', 'keywords', 'standard_name_vocabulary', 'references',
    'comment',
    'group_name', 'climate_forcing', 'scenario', 'location',
]

# Fields eligible for auto-copy from an input file's global attributes.
# 'id' is excluded: it's meant to uniquely identify each output product, so
# it should come from the YAML/DEFAULTS rather than be inherited unchanged
# from an upstream file.
_COPYABLE_KEYS = [k for k in DEFAULTS if k != 'id']


def extract_known_attrs(ds):
    """
    Pull whatever global attributes on an already-open netCDF4.Dataset match
    our known metadata fields (e.g. an upstream ADCIRC fort.63.nc, or a
    compact file written by an earlier pipeline step).

    Parameters
    ----------
    ds : netCDF4.Dataset

    Returns
    -------
    dict, only the recognized, non-empty fields found.
    """
    found = {}
    for key in _COPYABLE_KEYS:
        if key in ds.ncattrs():
            value = getattr(ds, key)
            if value not in (None, ''):
                found[key] = value
    return found


def read_known_attrs(path):
    """Convenience wrapper: open `path` read-only and call extract_known_attrs."""
    ds = nc.Dataset(str(path), 'r')
    try:
        return extract_known_attrs(ds)
    finally:
        ds.close()


def load_metadata(yaml_path=None, source_attrs=None, cli_overrides=None):
    """
    Assemble the global metadata dict for an output file.

    Precedence (lowest to highest): built-in DEFAULTS, then fields copied
    from `source_attrs` (e.g. an input file's existing global attributes),
    then fields explicitly set in the YAML file, then `cli_overrides`. This
    means metadata already present on an input file propagates forward
    automatically when no YAML is given, a YAML file always wins over an
    inherited value, and an explicit CLI flag always wins over the YAML
    (e.g. --group-name overriding "group_name" from --metadata-yaml).

    Parameters
    ----------
    yaml_path : str or Path or None
        Path to a YAML file (see metadata_template.yaml).
    source_attrs : dict or None
        Fields to inherit from an input file, typically the result of
        extract_known_attrs()/read_known_attrs() on that file.
    cli_overrides : dict or None
        Fields explicitly passed on the command line, e.g. from
        naming_overrides_from_args(). None/empty-string values are ignored.

    Returns
    -------
    dict
    """
    metadata = dict(DEFAULTS)

    if source_attrs:
        for key, value in source_attrs.items():
            if key in metadata and value not in (None, ''):
                metadata[key] = value

    if yaml_path is not None:
        with open(yaml_path) as f:
            user_fields = yaml.safe_load(f) or {}
        for key, value in user_fields.items():
            if value is not None and value != '':
                metadata[key] = value

    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None and value != '':
                metadata[key] = value

    return metadata


def add_naming_args(parser):
    """
    Add the --group-name/--climate-forcing/--scenario/--location CLI flags
    (shared filename-naming-convention fields) to an argparse parser.
    """
    parser.add_argument(
        '--group-name',
        help='Overrides "group_name" from --metadata-yaml, e.g. "Argonne". '
             'Required (via CLI or YAML) to build output filenames.',
    )
    parser.add_argument(
        '--climate-forcing',
        help='Overrides "climate_forcing" from --metadata-yaml, e.g. '
             '"CFSv2". Required (via CLI or YAML) to build output filenames.',
    )
    parser.add_argument(
        '--scenario',
        help='Overrides "scenario" from --metadata-yaml, e.g. "Reanalysis". '
             'Required (via CLI or YAML) to build output filenames.',
    )
    parser.add_argument(
        '--location',
        help='Overrides "location" from --metadata-yaml, e.g. "GESLA". '
             'Required (via CLI or YAML) to build output filenames.',
    )


def naming_overrides_from_args(args):
    """Extract the add_naming_args() flags from a parsed args namespace."""
    return {
        'group_name': args.group_name,
        'climate_forcing': args.climate_forcing,
        'scenario': args.scenario,
        'location': args.location,
    }


def _format_time_range(year_or_range):
    """
    Format a year or year range as a CMIP6-style YYYYMM-YYYYMM time range.

    All SurgeMIP outputs span full calendar years, so the month is always
    01 for the start and 12 for the end (e.g. 2000 -> '200001-200012',
    '1947-2025' -> '194701-202512').
    """
    start, _, end = str(year_or_range).partition('-')
    end = end or start
    return f'{start}01-{end}12'


def discover_hourly_year_files(hourly_dir, var_name):
    """
    Find per-year hourly files for `var_name` (e.g. 'twl' or 'ssgh') written
    by build_filename() in hourly_dir, sorted by year.

    Matches the current build_filename() convention for a single-year
    Hourly file: '{var_name}_Hourly_..._{YYYY}01-{YYYY}12.nc' (a Hourly file
    always spans exactly one calendar year, so the two YYYY tokens are
    equal; only the start year is captured/returned).
    """
    hourly_dir = Path(hourly_dir)
    pattern = re.compile(
        rf'^{re.escape(var_name)}_Hourly_.*_(\d{{4}})01-\d{{4}}12\.nc$')
    results = []
    for f in sorted(hourly_dir.glob(f'{var_name}_Hourly_*.nc')):
        m = pattern.search(f.name)
        if m:
            results.append((int(m.group(1)), f))
    results.sort(key=lambda x: x[0])
    return results


def build_filename(metadata, timestep, variable_key, year_or_range):
    """
    Build an output filename following the SurgeMIP naming convention,
    modeled on CMIP6's `variable_id_table_id_source_id_experiment_id_
    variant_label_grid_label_time-range.nc` pattern (see
    https://help.ceda.ac.uk/article/4801-cmip6-data): the variable leads,
    the time range trails, and provenance/descriptor tokens sit in between.

        Variable_TimeStep_GroupName_ClimateForcing_Scenario_Location_TimeRange.nc

    e.g. twl_Hourly_Argonne_CFSv2_Reanalysis_GESLA_200001-200012.nc

    Parameters
    ----------
    metadata : dict
        Result of load_metadata(); must have non-empty group_name,
        climate_forcing, scenario, and location fields.
    timestep : str
        e.g. 'Hourly' or 'MonthlyMax'.
    variable_key : str
        'WaterLevel' or 'StormSurge' — looked up in VARIABLES for the
        filename's short variable token (twl / ssgh).
    year_or_range : int or str
        e.g. 2000 for a single-year hourly file, or '1947-2025' for a
        full-period monthly-max file.

    Returns
    -------
    str
    """
    missing = [f for f in NAMING_FIELDS if not metadata.get(f)]
    if missing:
        raise ValueError(
            f'Missing required naming field(s) {missing}. Set via '
            f'--metadata-yaml or the corresponding --group-name/'
            f'--climate-forcing/--scenario/--location CLI flag.')

    bad = [f for f in NAMING_FIELDS if '_' in str(metadata[f])]
    if bad:
        raise ValueError(
            f'Naming field(s) {bad} contain an underscore, which is the '
            f'filename field separator — use a token without underscores.')

    if variable_key not in VARIABLES:
        raise ValueError(f'Unknown variable {variable_key!r}, expected one '
                         f'of {list(VARIABLES)}')

    var_name = VARIABLES[variable_key]['name']
    time_range = _format_time_range(year_or_range)
    parts = [var_name, timestep]
    parts += [metadata[f] for f in NAMING_FIELDS]
    parts += [time_range]
    return '_'.join(parts) + '.nc'


def set_global_attrs(ds, metadata, *, title, summary, feature_type='timeSeries',
                     extra=None):
    """
    Write CF/ACDD-style global attributes to a freshly-created netCDF4 Dataset.

    Parameters
    ----------
    ds : netCDF4.Dataset
        Dataset opened in write mode.
    metadata : dict
        Result of load_metadata().
    title : str
        Per-file title (differs between the hourly, monthly, and block
        maxima outputs), not part of the shared YAML template.
    summary : str
        Per-file summary/abstract.
    feature_type : str
        CF featureType attribute (default 'timeSeries').
    extra : dict or None
        Additional pipeline-specific attributes (e.g. source_csv,
        detiding_method) written after the common block.
    """
    ds.Conventions = 'CF-1.8'
    ds.featureType = feature_type
    ds.Metadata_Conventions = 'Unidata Dataset Discovery v1.0'
    ds.title = title
    ds.summary = summary
    ds.date_created = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    ds.history = ''

    for key in _ATTR_ORDER:
        setattr(ds, key, metadata.get(key, DEFAULTS.get(key, '')))

    if extra:
        for key, value in extra.items():
            setattr(ds, key, value)


def set_geospatial_extent(ds, lon, lat, depth=None, positive='down'):
    """
    Set geospatial_* global attributes describing the station coordinates.

    Parameters
    ----------
    ds : netCDF4.Dataset
    lon, lat : array-like
        Station longitudes/latitudes (degrees).
    depth : array-like or None
        Station depths/elevations, if available.
    positive : str
        'down' if depth increases downward (ADCIRC convention), 'up' for
        elevation. Mirrors the variable-level `positive` attribute.
    """
    ds.geospatial_lat_min = float(min(lat))
    ds.geospatial_lat_max = float(max(lat))
    ds.geospatial_lat_units = 'degrees_north'
    ds.geospatial_lat_resolution = 'point'
    ds.geospatial_lon_min = float(min(lon))
    ds.geospatial_lon_max = float(max(lon))
    ds.geospatial_lon_units = 'degrees_east'
    ds.geospatial_lon_resolution = 'point'

    if depth is not None:
        ds.geospatial_vertical_min = float(min(depth))
        ds.geospatial_vertical_max = float(max(depth))
        ds.geospatial_vertical_units = 'm'
        ds.geospatial_vertical_positive = positive


def read_times(ds, varname='time'):
    """
    Read a time variable as real timestamps, using whatever `units` and
    `calendar` it actually declares — not an assumed epoch. This matters
    for files written before the pipeline's reference epoch changed, or any
    file that otherwise uses a different "<units> since <epoch>" string.

    Parameters
    ----------
    ds : netCDF4.Dataset
    varname : str

    Returns
    -------
    pandas.DatetimeIndex
    """
    var = ds.variables[varname]
    calendar = getattr(var, 'calendar', 'standard')
    cftime_dates = nc.num2date(var[:], var.units, calendar)
    try:
        return pd.to_datetime([d.isoformat() for d in cftime_dates])
    except Exception:
        return pd.to_datetime([d.isoformat() for d in cftime_dates], format='ISO8601')


def write_times(ds, varname, times):
    """
    Convert `times` into the numeric values of an existing time variable,
    honoring whatever `units`/`calendar` that variable already declares
    (rather than assuming the pipeline's current reference epoch). This
    keeps appends self-consistent with a file created under a different
    epoch, e.g. an older run using "hours since 1970-01-01".

    Parameters
    ----------
    ds : netCDF4.Dataset
    varname : str
    times : pandas.DatetimeIndex or sequence of datetime-like

    Returns
    -------
    ndarray of float, ready to assign into ds.variables[varname][...]
    """
    var = ds.variables[varname]
    calendar = getattr(var, 'calendar', 'standard')
    times = pd.DatetimeIndex(times)
    return nc.date2num(times.to_pydatetime(), var.units, calendar)


def update_time_coverage(ds, times):
    """
    Set or extend the time_coverage_start / time_coverage_end global
    attributes to cover `times`, in addition to whatever the dataset
    already covers (so repeated appends across years stay correct).

    Parameters
    ----------
    ds : netCDF4.Dataset
        Dataset opened in append/write mode.
    times : pandas.DatetimeIndex or sequence of datetime-like
        Timestamps newly written in this call.
    """
    new_start = min(times)
    new_end = max(times)

    existing_start = getattr(ds, 'time_coverage_start', '')
    existing_end = getattr(ds, 'time_coverage_end', '')

    fmt = '%Y-%m-%d %H:%M:%S'
    start_str = new_start.strftime(fmt)
    end_str = new_end.strftime(fmt)

    if not existing_start or start_str < existing_start:
        ds.time_coverage_start = start_str
    if not existing_end or end_str > existing_end:
        ds.time_coverage_end = end_str

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
(geospatial extent, time coverage, creation_date) are set separately by
set_geospatial_extent() and update_time_coverage(), since they can't come
from a static YAML file or be copied from an input file.
"""

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import netCDF4 as nc
import numpy as np
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
    'contact': '',
    'creator_name': '',
    'creator_id': '',
    'creator_email': '',
    'researcher_name': '',
    'researcher_id': '',
    'researcher_email': '',
    'researcher_affiliation': '',
    'source': 'ADCIRC unstructured-mesh storm surge model (STOFS2D-Global '
              'setup) forced by CFS reanalysis atmospheric fields',
    'forcing': '',
    # Horizontal CRS of the source model/station coordinates (ACDD
    # `geospatial_bounds_crs`, e.g. an EPSG code), written on the
    # geospatial_bounds bounding box by set_geospatial_extent(). WGS84
    # lat/lon is assumed throughout this pipeline (see the fixed WGS84
    # ellipsoid parameters on the `crs` variable in write_node_block()), so
    # this should only be overridden if that ever changes.
    'geospatial_bounds_crs': 'EPSG:4326',
    # Vertical datum/CRS of `node_depth` (ACDD
    # `geospatial_bounds_vertical_crs`, e.g. an EPSG code or geoid/datum
    # name), written alongside geospatial_bounds_crs. No single value is
    # valid across all runs — ADCIRC depth is geoid-referenced, but the
    # specific geoid model varies by mesh/campaign — so this is left blank
    # unless set via --metadata-yaml. See set_geospatial_extent().
    'geospatial_bounds_vertical_crs': '',
    # Vertical datum of the water level variable's own values (e.g. "LMSL"
    # or "NAVD88") — distinct from node_depth's vertical datum above.
    # Written as that variable's `datum` attribute (see resolve_datum()).
    # Left blank here since the right default depends on which variable is
    # being written: 'LMSL' for twl, blank for ssgh (a detided residual
    # with no fixed absolute vertical reference) — see
    # VARIABLES['default_vertical_datum'].
    'datum': '',
    'license': '',
    'acknowledgment': '',
    'references': '',
    'comment': '',
    'sea_name': 'global',
    'keywords': 'storm surge; coastal flooding; water level; ADCIRC; SurgeMIP',
    'standard_name_vocabulary': 'CF Standard Name Table',
    # CMIP6-style global attributes (see
    # http://goo.gl/v1drZl, "CMIP6 Global Attributes, DRS, Filenames,
    # Directory Structure, and CV's") that don't already have a SurgeMIP
    # equivalent below. realm/further_info_url are free text here rather
    # than drawn from the official CMIP6 controlled vocabularies, since
    # SurgeMIP isn't a registered CMIP6 activity.
    'realm': 'ocean',
    'further_info_url': '',
    # Short, filename-safe tokens used to build the SurgeMIP output filename
    # convention (see NAMING_FIELDS/build_filename below), and — via
    # set_global_attrs() — the CMIP6-style institution_id/source_id/
    # experiment_id global attributes. Distinct from the free-text
    # institution/forcing/source provenance fields above.
    'group_name': '',
    'climate_forcing': '',
    'scenario': '',
    'location': '',
}

# Canonical variable definitions for the SurgeMIP naming/CF-attribute
# convention. 'name' is both the netCDF variable name and the filename
# token; 'standard_name' is written as the variable's standard_name
# attribute even though these two terms are not (yet) part of the official
# CF Standard Name Table. 'default_vertical_datum' is this variable's
# default `datum` attribute (see resolve_datum()): twl is referenced to
# local mean sea level by default, while ssgh (a detided residual) has no
# fixed absolute vertical reference.
VARIABLES = {
    'WaterLevel': {
        'name': 'twl',
        'standard_name': 'total_water_level',
        'long_name': 'Total Water Level',
        'default_vertical_datum': 'LMSL',
    },
    'StormSurge': {
        'name': 'ssgh',
        'standard_name': 'storm_surge_height',
        'long_name': 'Storm Surge Height',
        'default_vertical_datum': '',
    },
}


def resolve_datum(metadata, variable_key):
    """
    Resolve the vertical datum for a variable's own values (e.g. 'LMSL' for
    twl), written as that variable's `datum` attribute by the pipeline's
    write functions. An explicit metadata['datum'] (set via
    --metadata-yaml) always wins; otherwise falls back to this variable's
    default (see VARIABLES). Returns '' if neither applies (e.g. ssgh by
    default), in which case callers should omit the attribute.
    """
    return metadata.get('datum') or VARIABLES[variable_key].get(
        'default_vertical_datum', '')

# Fields required to build the CMIP6-style
# Variable_Frequency_GroupName_ClimateForcing_Scenario_Location_TimeRange
# filename convention (see build_filename), in the order they appear
# between the Frequency and TimeRange tokens.
NAMING_FIELDS = ['group_name', 'climate_forcing', 'scenario', 'location']

# CMIP6 "frequency" CV token (http://goo.gl/v1drZl) for each SurgeMIP
# timestep. Used both as the filename's Frequency token (build_filename)
# and to derive the `frequency` global attribute (set_global_attrs);
# extend this if a new timestep is introduced.
_TIMESTEP_TO_FREQUENCY = {
    'Hourly': '1hr',
    'DailyMax': 'day',
    'MonthlyMax': 'mon',
}

# Order in which global attributes are written, matching the style of the
# GTSMv3 reanalysis reference file this template is based on, with CMIP6-
# style attributes (institution_id, source_id, experiment_id, realm,
# product, frequency, table_id, variable_id — see set_global_attrs) placed
# near their Table-1 counterparts at http://goo.gl/v1drZl. institution_id/
# source_id/experiment_id are computed from group_name/climate_forcing/
# scenario rather than listed separately here, to avoid writing the same
# value under two different attribute names.
_ATTR_ORDER = [
    'id', 'project', 'acknowledgment', 'contact',
    'creator_name', 'creator_id', 'creator_email',
    'researcher_name', 'researcher_id', 'researcher_email',
    'researcher_affiliation',
    'license', 'institution', 'institution_id', 'further_info_url',
    'sea_name', 'source', 'source_id', 'forcing', 'experiment_id',
    'realm', 'product', 'frequency', 'table_id', 'variable_id', 'location',
    'keywords', 'standard_name_vocabulary', 'references', 'comment',
]

# Fields eligible for auto-copy from an input file's global attributes.
# 'id' is excluded: it's meant to uniquely identify each output product, so
# it should come from the YAML/DEFAULTS rather than be inherited unchanged
# from an upstream file. 'datum' and 'geospatial_bounds_crs'/
# 'geospatial_bounds_vertical_crs' are also excluded: they're not written
# via the generic _ATTR_ORDER loop (see resolve_datum()/
# set_geospatial_extent()), but they're still useful as config defaults —
# read via metadata.get(...) directly rather than through this list.
_COPYABLE_KEYS = [k for k in DEFAULTS if k not in (
    'id', 'datum', 'geospatial_bounds_crs', 'geospatial_bounds_vertical_crs')]


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

    # 'forcing' (free-text) and 'climate_forcing' (the short naming-
    # convention token used to build filenames and the CMIP6-style
    # source_id attribute) usually name the same thing. If the user never
    # set 'forcing' explicitly (YAML/source file/CLI), default it to
    # whatever climate_forcing resolved to, rather than leaving it blank.
    if not metadata.get('forcing') and metadata.get('climate_forcing'):
        metadata['forcing'] = metadata['climate_forcing']

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
    Format a year, year range, or month-precision range as a CMIP6-style
    YYYYMM-YYYYMM time range.

    Most SurgeMIP outputs span full calendar years, so a bare year or
    year-year range gets month 01 appended to the start and 12 to the end
    (e.g. 2000 -> '200001-200012', '1947-2025' -> '194701-202512'). A
    caller that already has month precision (e.g. a --start-date/
    --end-date-clipped partial-year file) can instead pass an already-built
    YYYYMM-YYYYMM string directly, which is passed through unchanged.
    """
    start, _, end = str(year_or_range).partition('-')
    end = end or start
    if len(start) == 6 and len(end) == 6:
        return f'{start}-{end}'
    return f'{start}01-{end}12'


def discover_hourly_year_files(hourly_dir, var_name):
    """
    Find per-year hourly files for `var_name` (e.g. 'twl' or 'ssgh') written
    by build_filename() in hourly_dir, sorted by year.

    Matches the current build_filename() convention for a single-year
    Hourly file: '{var_name}_1hr_..._{YYYY}01-{YYYY}12.nc' (a Hourly file
    always spans exactly one calendar year, so the two YYYY tokens are
    equal; only the start year is captured/returned).
    """
    hourly_dir = Path(hourly_dir)
    freq = _TIMESTEP_TO_FREQUENCY['Hourly']
    pattern = re.compile(
        rf'^{re.escape(var_name)}_{re.escape(freq)}_.*_(\d{{4}})01-\d{{4}}12\.nc$')
    results = []
    for f in sorted(hourly_dir.glob(f'{var_name}_{freq}_*.nc')):
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
    The Frequency token is the CMIP6 CV frequency abbreviation (see
    _TIMESTEP_TO_FREQUENCY), not the raw `timestep` value.

        Variable_Frequency_GroupName_ClimateForcing_Scenario_Location_TimeRange.nc

    e.g. twl_1hr_Argonne_CFSv2_Reanalysis_GESLA_200001-200012.nc

    Parameters
    ----------
    metadata : dict
        Result of load_metadata(); must have non-empty group_name,
        climate_forcing, scenario, and location fields.
    timestep : str
        e.g. 'Hourly' or 'MonthlyMax' — looked up in _TIMESTEP_TO_FREQUENCY
        for the filename's Frequency token (e.g. '1hr' / 'mon').
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

    if timestep not in _TIMESTEP_TO_FREQUENCY:
        raise ValueError(f'Unknown timestep {timestep!r}, expected one of '
                         f'{list(_TIMESTEP_TO_FREQUENCY)}')

    var_name = VARIABLES[variable_key]['name']
    frequency = _TIMESTEP_TO_FREQUENCY[timestep]
    time_range = _format_time_range(year_or_range)
    parts = [var_name, frequency]
    parts += [metadata[f] for f in NAMING_FIELDS]
    parts += [time_range]
    return '_'.join(parts) + '.nc'


def set_global_attrs(ds, metadata, *, title, summary, timestep, variable_key,
                     feature_type='timeSeries', extra=None):
    """
    Write CF/ACDD- and CMIP6-style global attributes to a freshly-created
    netCDF4 Dataset. The CMIP6-style attributes follow the naming/format
    conventions in http://goo.gl/v1drZl ("CMIP6 Global Attributes, DRS,
    Filenames, Directory Structure, and CV's"), adapted for a non-CMIP
    dataset: only the generic, broadly-applicable attributes are included
    (institution_id, source_id, experiment_id, realm, product, frequency,
    table_id, variable_id, tracking_id, creation_date). CMIP-ensemble/DRS
    bookkeeping fields that assume registered CMIP6 controlled vocabularies
    (mip_era, activity_id, data_specs_version, parent_*, branch_*,
    realization/initialization/physics/forcing_index, sub_experiment_id,
    variant_label, grid_label) are omitted, since SurgeMIP isn't a
    registered CMIP6 activity and setting them (e.g. mip_era="CMIP6") would
    misrepresent this as literal CMIP6 output.

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
    timestep : str
        e.g. 'Hourly' or 'MonthlyMax' — the same value passed to
        build_filename(). Written as the `table_id` attribute and used to
        derive `frequency`.
    variable_key : str
        'WaterLevel' or 'StormSurge' — the same value passed to
        build_filename(). Written as the `variable_id` attribute.
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
    ds.creation_date = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    ds.tracking_id = f'hdl:21.14100/{uuid.uuid4()}'
    ds.history = ''

    computed = {
        'institution_id': metadata.get('group_name', ''),
        'source_id': metadata.get('climate_forcing', ''),
        'experiment_id': metadata.get('scenario', ''),
        'variable_id': VARIABLES[variable_key]['name'],
        'table_id': timestep,
        'frequency': _TIMESTEP_TO_FREQUENCY.get(timestep, ''),
        'product': 'model-output',
    }

    for key in _ATTR_ORDER:
        setattr(ds, key, computed.get(key, metadata.get(key, DEFAULTS.get(key, ''))))

    if extra:
        for key, value in extra.items():
            setattr(ds, key, value)


def set_geospatial_extent(ds, lon, lat, depth=None, positive='down',
                          crs='EPSG:4326', vertical_crs=''):
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
    crs : str
        ACDD `geospatial_bounds_crs` value describing the horizontal CRS of
        `lon`/`lat` (and of the source model's own coordinates, since this
        pipeline doesn't reproject). Default 'EPSG:4326' (WGS84), matching
        the fixed WGS84 ellipsoid parameters on the `crs` variable written
        by write_node_block() — override via metadata['geospatial_bounds_crs']
        only if that ever changes.
    vertical_crs : str
        ACDD `geospatial_bounds_vertical_crs` value describing the vertical
        reference for `depth` (e.g. an EPSG code or geoid/datum name for the
        specific model run's vertical datum). There's no single value valid
        across all runs (ADCIRC depth is geoid-referenced, but the geoid
        model varies), so this is left to the caller/metadata YAML; the
        attribute is only written when non-empty.

    Notes
    -----
    `geospatial_bounds_crs`/`geospatial_bounds_vertical_crs` are, per ACDD
    1.3, specifically the CRS of the `geospatial_bounds` WKT geometry (not
    of the plain `geospatial_lat/lon/vertical_min/max` attributes above) —
    so a minimal bounding-box `geospatial_bounds` polygon is written here to
    ground them per spec, rather than setting the CRS attributes alone.
    This is also where the source model's own horizontal/vertical CRS is
    recorded — there's no separate "source_crs" attribute elsewhere.
    """
    lat_min, lat_max = float(min(lat)), float(max(lat))
    lon_min, lon_max = float(min(lon)), float(max(lon))

    ds.geospatial_lat_min = lat_min
    ds.geospatial_lat_max = lat_max
    ds.geospatial_lat_units = 'degrees_north'
    ds.geospatial_lat_resolution = 'point'
    ds.geospatial_lon_min = lon_min
    ds.geospatial_lon_max = lon_max
    ds.geospatial_lon_units = 'degrees_east'
    ds.geospatial_lon_resolution = 'point'
    ds.geospatial_bounds = (
        f'POLYGON(({lon_min} {lat_min}, {lon_max} {lat_min}, '
        f'{lon_max} {lat_max}, {lon_min} {lat_max}, '
        f'{lon_min} {lat_min}))'
    )
    ds.geospatial_bounds_crs = crs

    if depth is not None:
        ds.geospatial_vertical_min = float(min(depth))
        ds.geospatial_vertical_max = float(max(depth))
        ds.geospatial_vertical_units = 'm'
        ds.geospatial_vertical_positive = positive
        if vertical_crs:
            ds.geospatial_bounds_vertical_crs = vertical_crs


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


# ---------------------------------------------------------------------------
# Model-dependent node indexing
# ---------------------------------------------------------------------------

# Registry of on-disk node-indexing conventions, keyed by the value stored
# in a file's `model_name` global attribute. 'dims' is 1 for a flat
# unstructured-mesh index (a single 'node_index' variable) or 2 for a
# structured/curvilinear grid (separate 'node_i'/'node_j' variables).
# 'base' is the on-disk numbering origin (0 or 1) used by that model's
# native node/cell numbering; internally, this pipeline always tracks
# indices 0-based and converts to/from `base` only in write_node_index()/
# read_node_index(). Only 'ADCIRC' is actually produced today (by
# extract_outputs_to_shoreline_pts.py, an unstructured mesh with 1-based
# fort.63.nc node numbering) — add an entry here for any other source
# model, e.g. a 0-based unstructured model or a 2-D structured grid:
#   'SCHISM': {'dims': 1, 'base': 0},
#   'STRUCTURED': {'dims': 2, 'base': 0},
NODE_INDEX_SCHEMES = {
    'ADCIRC': {'dims': 1, 'base': 1},
}
DEFAULT_MODEL_NAME = 'ADCIRC'


def get_node_index_scheme(model_name):
    """Look up the on-disk node-indexing convention for `model_name`."""
    if model_name not in NODE_INDEX_SCHEMES:
        raise ValueError(
            f'Unknown model_name {model_name!r}; add an entry to '
            f'NODE_INDEX_SCHEMES in nc_metadata.py, expected one of '
            f'{list(NODE_INDEX_SCHEMES)}')
    return NODE_INDEX_SCHEMES[model_name]


def model_node_long_name(model_name, what):
    """e.g. model_node_long_name('ADCIRC', 'Longitude') -> 'Longitude of
    matched ADCIRC mesh node'."""
    return f'{what} of matched {model_name} mesh node'


def write_node_index(ds, model_name, node_index, dim='node'):
    """
    Write the node-identifying index variable(s) using the on-disk
    convention registered for `model_name` (see NODE_INDEX_SCHEMES),
    converting from the pipeline's internal 0-based index to that model's
    native base, and records `model_name` as a global attribute so
    read_node_index() can reverse the conversion later.

    Parameters
    ----------
    ds : netCDF4.Dataset
        Dataset opened in write mode, with dimension `dim` already created.
    model_name : str
        Key into NODE_INDEX_SCHEMES.
    node_index : array-like
        0-based index into the source model's node/cell array. Shape (n,)
        for a 1-D ('dims': 1) scheme, or (n, 2) of (i, j) pairs for a 2-D
        ('dims': 2) scheme.
    dim : str
        Name of the existing netCDF dimension these variables are defined
        over (default 'node').
    """
    scheme = get_node_index_scheme(model_name)
    base = scheme['base']
    ds.model_name = model_name

    if scheme['dims'] == 1:
        node_index = np.asarray(node_index)
        v = ds.createVariable('node_index', 'i4', (dim,), zlib=True, complevel=1)
        v.long_name = (f'{base}-based index, matches original {model_name} '
                       f'model node numbering')
        v.cf_role = 'timeseries_id'
        v[:] = node_index + base
    else:
        node_index = np.asarray(node_index)
        if node_index.ndim != 2 or node_index.shape[1] != 2:
            raise ValueError(
                f'{model_name} uses 2-D (i, j) indexing; node_index must '
                f'have shape (n, 2), got {node_index.shape}')
        v = ds.createVariable('node_i', 'i4', (dim,), zlib=True, complevel=1)
        v.long_name = (f'{base}-based i-index, matches original {model_name} '
                       f'model node numbering')
        v.cf_role = 'timeseries_id'
        v[:] = node_index[:, 0] + base

        v = ds.createVariable('node_j', 'i4', (dim,), zlib=True, complevel=1)
        v.long_name = (f'{base}-based j-index, matches original {model_name} '
                       f'model node numbering')
        v[:] = node_index[:, 1] + base


def read_node_index(ds, dim='node'):
    """
    Read whichever node-index variable(s) an input file has, converting
    back to the pipeline's internal 0-based convention using the on-disk
    base registered for its `model_name` global attribute (defaulting to
    'ADCIRC' for files written before model_name was tracked).

    Returns
    -------
    model_name : str
    node_index : ndarray
        0-based; shape (n,) for a 1-D scheme or (n, 2) of (i, j) pairs for
        a 2-D scheme.
    """
    model_name = getattr(ds, 'model_name', DEFAULT_MODEL_NAME)
    scheme = get_node_index_scheme(model_name)
    base = scheme['base']
    if scheme['dims'] == 1:
        node_index = np.array(ds.variables['node_index'][:], dtype=np.int64) - base
    else:
        i_idx = np.array(ds.variables['node_i'][:], dtype=np.int64) - base
        j_idx = np.array(ds.variables['node_j'][:], dtype=np.int64) - base
        node_index = np.stack([i_idx, j_idx], axis=1)
    return model_name, node_index


def write_node_block(ds, model_name, node, dim='node', point_set_source='GSHHS'):
    """
    Write the full set of static per-node variables shared by every
    SurgeMIP pipeline output: crs, node_index (see write_node_index),
    node_lon/lat/depth, point_lon/lat, dist_km.

    Parameters
    ----------
    ds : netCDF4.Dataset
        Dataset opened in write mode, with dimension `dim` already created.
    model_name : str
        Key into NODE_INDEX_SCHEMES.
    node : dict
        'node_index' (see write_node_index for shape/convention),
        'node_lon', 'node_lat', 'node_depth', 'point_lon', 'point_lat',
        'dist_km' : ndarray (n,)
    dim : str
    point_set_source : str
        Short label for the source CSV point set (e.g. 'GSHHS', 'GESLA'),
        written parenthetically in the point_lon/point_lat long_name.
    """
    crs_v = ds.createVariable('crs', 'i4')
    crs_v.grid_mapping_name = 'latitude_longitude'
    crs_v.longitude_of_prime_meridian = 0.0
    crs_v.semi_major_axis = 6378137.0
    crs_v.inverse_flattening = 298.257223563
    crs_v[:] = -2147483647

    write_node_index(ds, model_name, node['node_index'], dim=dim)

    for name, key, std_name, long_name, units in [
        ('node_lon', 'node_lon', 'longitude',
         model_node_long_name(model_name, 'Longitude'), 'degrees_east'),
        ('node_lat', 'node_lat', 'latitude',
         model_node_long_name(model_name, 'Latitude'), 'degrees_north'),
        ('node_depth', 'node_depth', 'sea_floor_depth_below_geoid',
         'Depth of matched mesh node below geoid', 'm'),
        ('point_lon', 'point_lon', 'longitude',
         f'Original CSV point longitude ({point_set_source})', 'degrees_east'),
        ('point_lat', 'point_lat', 'latitude',
         f'Original CSV point latitude ({point_set_source})', 'degrees_north'),
    ]:
        v = ds.createVariable(name, 'f8', (dim,), zlib=True, complevel=1)
        v.standard_name = std_name
        v.long_name = long_name
        v.units = units
        v[:] = node[key]
        if name == 'node_depth':
            v.positive = 'down'

    v = ds.createVariable('dist_km', 'f4', (dim,), zlib=True, complevel=1)
    v.long_name = 'Distance from CSV point to matched mesh node'
    v.units = 'km'
    v[:] = node['dist_km'].astype(np.float32)


def read_node_block(ds, dim='node'):
    """
    Read the full set of static per-node variables written by
    write_node_block().

    Returns
    -------
    dict with 'model_name', 'node_index' (0-based; see read_node_index),
    'node_lon', 'node_lat', 'node_depth', 'point_lon', 'point_lat',
    'dist_km', and 'source_csv' (from the `source_csv` global attribute, if
    present).
    """
    model_name, node_index = read_node_index(ds, dim=dim)
    return dict(
        model_name=model_name,
        node_index=node_index,
        node_lon=np.array(ds.variables['node_lon'][:], dtype=np.float64),
        node_lat=np.array(ds.variables['node_lat'][:], dtype=np.float64),
        node_depth=np.array(ds.variables['node_depth'][:], dtype=np.float64),
        point_lon=np.array(ds.variables['point_lon'][:], dtype=np.float64),
        point_lat=np.array(ds.variables['point_lat'][:], dtype=np.float64),
        dist_km=np.array(ds.variables['dist_km'][:], dtype=np.float64),
        source_csv=getattr(ds, 'source_csv', ''),
    )

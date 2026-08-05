#!/usr/bin/env python3
"""
Shared CF/ACDD-style global metadata for the SurgeMIP pipeline's NetCDF
outputs.

User-facing fields (institution, contact, project, license, ...) come from
a YAML file — see metadata_template.yaml for the editable template and a
description of each field. Fields the user leaves out fall back to whatever
matching global attributes already exist on the input file being read (e.g.
an upstream fort.63.nc, or a compact file from an earlier pipeline step),
then to the DEFAULTS below. Fields that are computed from the data itself
(geospatial extent, time coverage, date_created) are set separately by
set_geospatial_extent() and update_time_coverage(), since they can't come
from a static YAML file or be copied from an input file.
"""

from datetime import datetime, timezone

import netCDF4 as nc
import yaml

# Fallback values used for any field not present in the user's YAML file.
DEFAULTS = {
    'id': 'SurgeMIP_shoreline_waterlevels',
    'project': 'SurgeMIP',
    'institution': 'Argonne National Laboratory',
    'contact': '',
    'source': 'ADCIRC STOFS2D-Global, CFS reanalysis atmospheric forcing',
    'license': '',
    'acknowledgment': '',
    'references': '',
    'comment': '',
    'sea_name': 'global',
    'keywords': 'storm surge; coastal flooding; water level; ADCIRC; SurgeMIP',
    'standard_name_vocabulary': 'CF Standard Name Table',
}

# Order in which global attributes are written, matching the style of the
# GTSMv3 reanalysis reference file this template is based on.
_ATTR_ORDER = [
    'id', 'project', 'acknowledgment', 'contact',
    'license', 'institution', 'sea_name', 'source', 'keywords',
    'standard_name_vocabulary', 'references',
    'comment',
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


def load_metadata(yaml_path=None, source_attrs=None):
    """
    Assemble the global metadata dict for an output file.

    Precedence (lowest to highest): built-in DEFAULTS, then fields copied
    from `source_attrs` (e.g. an input file's existing global attributes),
    then fields explicitly set in the YAML file. This means metadata already
    present on an input file propagates forward automatically when no YAML
    is given, but a YAML file always wins if the user wants to override it.

    Parameters
    ----------
    yaml_path : str or Path or None
        Path to a YAML file (see metadata_template.yaml).
    source_attrs : dict or None
        Fields to inherit from an input file, typically the result of
        extract_known_attrs()/read_known_attrs() on that file.

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

    return metadata


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

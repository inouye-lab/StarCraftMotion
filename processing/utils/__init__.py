"""Reference data for the StarCraft Motion dataset.

These modules are not consumed by the data-processing pipeline itself; they
are shipped as canonical references for downstream consumers of the dataset:

  * :mod:`utils.coarse_action_mapping` -- maps raw SC2 ``ability_id`` values
    (as stored in ``unit_data/repeated/ability_id``) to 12 coarse action
    classes suitable for action prediction.
  * :mod:`utils.unit_type_map` -- maps raw SC2 ``unit_type`` IDs to a dense
    contiguous index, with race and movement metadata.
"""

from .coarse_action_mapping import (
    ABILITY_ID_TO_COARSE_ACTION,
    COARSE_ACTION_NAMES,
    get_coarse_action,
    get_coarse_action_name,
)
from .unit_type_map import (
    MOVING_UNIT_TYPE_IDS,
    NUM_UNIT_TYPES,
    SC2_ID_TO_INDEX,
    STATIC_UNIT_TYPE_IDS,
    UNIT_ID_TO_NAME,
    UNIT_ID_TO_RACE,
    UNKNOWN_INDEX,
    describe_unit_type,
    get_unit_name,
    get_unit_race,
    is_moving_unit_type,
    is_static_unit_type,
    remap_unit_type,
)

__all__ = [
    "ABILITY_ID_TO_COARSE_ACTION",
    "COARSE_ACTION_NAMES",
    "MOVING_UNIT_TYPE_IDS",
    "NUM_UNIT_TYPES",
    "SC2_ID_TO_INDEX",
    "STATIC_UNIT_TYPE_IDS",
    "UNIT_ID_TO_NAME",
    "UNIT_ID_TO_RACE",
    "UNKNOWN_INDEX",
    "describe_unit_type",
    "get_coarse_action",
    "get_coarse_action_name",
    "get_unit_name",
    "get_unit_race",
    "is_moving_unit_type",
    "is_static_unit_type",
    "remap_unit_type",
]

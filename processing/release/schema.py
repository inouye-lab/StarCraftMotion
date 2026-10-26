"""Shared lossless mapping for the camera-ready StarCraft Motion release.

The original 37 fields retain their order and types. Integer attributes are
stored as int64 in HDF5 even where their Parquet representation is int32.
"""
import json
from pathlib import Path
import h5py
import numpy as np
import pyarrow as pa

ORIGINAL_SCHEMA = pa.schema([
    pa.field('split',            pa.string()),
    pa.field('map_name',         pa.string()),
    pa.field('replay_hash',      pa.string()),
    pa.field('segment_idx',      pa.int32()),
    pa.field('n_timesteps',      pa.int32()),
    pa.field('n_units',          pa.int32()),
    pa.field('map_T',            pa.int32()),
    pa.field('map_H',            pa.int32()),
    pa.field('map_W',            pa.int32()),
    # map data — flat list<bool>, reshape to (map_T, map_H, map_W)
    pa.field('map_creep',        pa.list_(pa.bool_())),
    pa.field('map_fow_p1',       pa.list_(pa.bool_())),
    pa.field('map_fow_p2',       pa.list_(pa.bool_())),
    # player economy — flat, reshape to (n_timesteps, 2)
    pa.field('food_cap',         pa.list_(pa.uint8())),
    pa.field('food_used',        pa.list_(pa.uint8())),
    pa.field('minerals',         pa.list_(pa.uint16())),
    pa.field('vespene',          pa.list_(pa.uint16())),
    # unit globals — length n_units
    pa.field('unit_owner',       pa.list_(pa.uint8())),
    pa.field('unit_tag',         pa.list_(pa.uint64())),
    # unit repeated — flat, reshape to (n_timesteps, n_units [, extra])
    pa.field('coordinate',       pa.list_(pa.float16())),  # (T, N, 3)
    pa.field('health',           pa.list_(pa.float16())),
    pa.field('health_max',       pa.list_(pa.float16())),
    pa.field('shield',           pa.list_(pa.float16())),
    pa.field('energy',           pa.list_(pa.float16())),
    pa.field('heading',          pa.list_(pa.float16())),
    pa.field('build_progress',   pa.list_(pa.float16())),
    pa.field('radius',           pa.list_(pa.float16())),
    pa.field('ability_id',       pa.list_(pa.uint32())),
    pa.field('unit_type',        pa.list_(pa.uint32())),
    pa.field('target_id',        pa.list_(pa.uint32())),
    pa.field('target_pos',       pa.list_(pa.float16())),  # (T, N, 2)
    pa.field('is_alive',         pa.list_(pa.bool_())),
    pa.field('is_burrowed',      pa.list_(pa.bool_())),
    pa.field('is_carried',       pa.list_(pa.bool_())),
    pa.field('is_flying',        pa.list_(pa.bool_())),
    pa.field('mineral_contents', pa.list_(pa.uint16())),
    pa.field('vespene_contents', pa.list_(pa.uint16())),
    pa.field('visible_status',   pa.list_(pa.uint8())),
])

# Per-column numpy dtype, used to build typed Arrow ListArrays without
# Python-level conversion. Must match the SCHEMA value-types above.
LIST_DTYPES = {
    'map_creep':        np.bool_,
    'map_fow_p1':       np.bool_,
    'map_fow_p2':       np.bool_,
    'food_cap':         np.uint8,
    'food_used':        np.uint8,
    'minerals':         np.uint16,
    'vespene':          np.uint16,
    'unit_owner':       np.uint8,
    'unit_tag':         np.uint64,
    'coordinate':       np.float16,
    'health':           np.float16,
    'health_max':       np.float16,
    'shield':           np.float16,
    'energy':           np.float16,
    'heading':          np.float16,
    'build_progress':   np.float16,
    'radius':           np.float16,
    'ability_id':       np.uint32,
    'unit_type':        np.uint32,
    'target_id':        np.uint32,
    'target_pos':       np.float16,
    'is_alive':         np.bool_,
    'is_burrowed':      np.bool_,
    'is_carried':       np.bool_,
    'is_flying':        np.bool_,
    'mineral_contents': np.uint16,
    'vespene_contents': np.uint16,
    'visible_status':   np.uint8,
}


KEY_COLUMNS = ['split', 'map_name', 'replay_hash', 'segment_idx']
SPLIT_OUT = {'train': 'train', 'val': 'validation', 'test': 'test'}
PADDING = 4
FRAME_ATTRIBUTES = ['chunk_start_frame', 'chunk_end_frame', 'map_frame_start',
                    'map_frame_skip', 'current_frame_idx']
ATTRIBUTE_COLUMNS = ['player_1_start_location', 'player_2_start_location',
                     'playable_area'] + FRAME_ATTRIBUTES + ['player_info', 'replay_info']
LIST_DTYPES.update(player_1_start_location=np.float32,
                   player_2_start_location=np.float32, playable_area=np.int64)
SCHEMA = pa.schema(list(ORIGINAL_SCHEMA) + [
    pa.field('player_1_start_location', pa.list_(pa.float32())),
    pa.field('player_2_start_location', pa.list_(pa.float32())),
    pa.field('playable_area', pa.list_(pa.int64())),
] + [pa.field(k, pa.int32()) for k in FRAME_ATTRIBUTES] + [
    pa.field('player_info', pa.string()), pa.field('replay_info', pa.string())])

# column -> (HDF5 dataset path, shape rule); no second copy in either converter.
DATASETS = {
    'map_creep': ('map_data/dynamic/creep', ('map_T', 'map_H', 'map_W')),
    'map_fow_p1': ('map_data/dynamic/fow_player_1', ('map_T', 'map_H', 'map_W')),
    'map_fow_p2': ('map_data/dynamic/fow_player_2', ('map_T', 'map_H', 'map_W')),
}
DATASETS.update({k: ('player_data/' + k, ('n_timesteps', 2))
                 for k in ['food_cap', 'food_used', 'minerals', 'vespene']})
DATASETS.update({k: ('unit_data/global/' + k, ('n_units',))
                 for k in ['unit_owner', 'unit_tag']})
DATASETS.update({f.name: ('unit_data/repeated/' + f.name,
                         ('n_timesteps', 'n_units') +
                         ((3,) if f.name == 'coordinate' else
                          (2,) if f.name == 'target_pos' else ()))
                 for f in list(ORIGINAL_SCHEMA)[18:]})
ATTRIBUTE_KEYS = set(ATTRIBUTE_COLUMNS) | {
    'chunk_idx', 'parent_replay', 'replay_name', 'map_name', 'map_size',
    'max_unique_units', 'total_frames'}
JSON_KEYS = {
    'player_info': {'player_id', 'race_requested', 'race_actual', 'player_result',
                    'player_mmr', 'player_apm'},
    'replay_info': {'game_duration_loops', 'game_duration_seconds', 'game_version',
                    'data_build', 'base_build'},
}


def key(row):
    return tuple(row[k] for k in KEY_COLUMNS)


def key_from_path(path):
    p = Path(path)
    replay, idx = p.stem.rsplit('_', 1)
    return p.parts[-3], p.parts[-2], replay, int(idx)


def relative_path(row):
    split, map_name, replay, idx = key(row)
    if split not in SPLIT_OUT or not 0 <= idx < 10 ** PADDING:
        raise ValueError(f'Invalid scenario key: {key(row)}')
    if any('/' in s or s in ('.', '..') for s in [map_name, replay]):
        raise ValueError(f'Invalid scenario key: {key(row)}')
    return Path(split) / map_name / f'{replay}_{idx:0{PADDING}d}.h5'


def array_shape(column, row):
    if column in DATASETS:
        return tuple(row[d] if isinstance(d, str) else d
                     for d in DATASETS[column][1])
    return (4 if column == 'playable_area' else 3,)


def expected_length(column, row):
    return int(np.prod(array_shape(column, row)))


def rebuild_attributes(row):
    attrs = {k: np.int64(row[k]) for k in FRAME_ATTRIBUTES}
    attrs.update({k: np.asarray(row[k], dtype=LIST_DTYPES[k]) for k in
                  ['player_1_start_location', 'player_2_start_location', 'playable_area']})
    attrs.update({k: row[k] for k in ['player_info', 'replay_info', 'map_name']})
    attrs.update(chunk_idx=np.int64(row['segment_idx']),
                 parent_replay=row['replay_hash'], replay_name=row['replay_hash'],
                 map_size=np.array([row['map_H'], row['map_W']], dtype=np.int64),
                 max_unique_units=np.int64(row['n_units']),
                 total_frames=np.int64(row['n_timesteps']))
    return attrs


def equal_values(a, b):
    """Exact typed values; floating bits must match except that NaN == NaN."""
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype.kind == 'f':
        bits = 'u' + str(a.dtype.itemsize)
        return bool(np.all((a.view(bits) == b.view(bits)) |
                           (np.isnan(a) & np.isnan(b))))
    return bool(np.array_equal(a, b))


def assert_source(f, row, path):
    """Fail closed on unrepresentable metadata or a broken redundancy rule."""
    def require(ok, rule):
        if not ok:
            raise ValueError(f'{path}: {rule}')
    require(set(f.attrs) == ATTRIBUTE_KEYS, 'attribute key set changed')
    require(row['n_timesteps'] == 145, 'total_frames != T != 145')
    require(row['segment_idx'] < 10 ** PADDING, 'segment index exceeds four digits')
    require(Path(path).stem == f"{row['replay_hash']}_{row['segment_idx']:04d}",
            'filename padding differs from four digits')
    require(row['current_frame_idx'] == 16, 'current_frame_idx != 16')
    require(row['map_frame_skip'] == 16, 'map_frame_skip != 16')
    for k in FRAME_ATTRIBUTES:
        require(np.iinfo(np.int32).min <= int(f.attrs[k]) <= np.iinfo(np.int32).max,
                f'{k} does not fit int32')
    for k in ['map_name', 'parent_replay', 'replay_name', 'player_info', 'replay_info']:
        require(isinstance(f.attrs[k], str), f'{k} must be str, not bytes')
    expected = rebuild_attributes(row)
    for k, v in expected.items():
        require(equal_values(f.attrs[k], v), f'attribute redundancy/dtype/shape: {k}')
        if isinstance(v, str):
            info = h5py.check_string_dtype(f.attrs.get_id(k).dtype)
            require(info is not None and info.encoding == 'utf-8' and info.length is None,
                    f'{k} is not variable-length UTF-8')
    found = set()
    f.visititems(lambda p, v: found.add(p) if isinstance(v, h5py.Dataset) else None)
    require(found == {p for p, _ in DATASETS.values()}, 'dataset path set changed')
    for column, (p, _) in DATASETS.items():
        require(f[p].shape == array_shape(column, row), f'dataset shape: {p}')
        require(f[p].dtype == np.dtype(LIST_DTYPES[column]), f'dataset dtype: {p}')
    for k, fields in JSON_KEYS.items():
        obj = json.loads(row[k])
        require(isinstance(obj, list if k == 'player_info' else dict), f'{k} JSON type')
        objs = obj if isinstance(obj, list) else [obj]
        require(all(not o or (isinstance(o, dict) and set(o) == fields) for o in objs),
                f'{k} JSON keys changed; audit before release')

"""Verify release keys, list offsets, sampled values, attributes and HF loading.

PyArrow must decode compressed Parquet pages. The all-row length check examines
only offsets, null bitmaps and scalar dimensions, never decoded list values.
Value comparisons and HDF5 reads are restricted to --keys (or all sample rows).
"""
import argparse
import collections
import json
import os
import subprocess
import sys
from multiprocessing import get_context
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from schema import (SCHEMA, ORIGINAL_SCHEMA, KEY_COLUMNS, LIST_DTYPES, DATASETS,
                    ATTRIBUTE_COLUMNS, FRAME_ATTRIBUTES, SPLIT_OUT, array_shape,
                    equal_values, key, key_from_path, relative_path)
from io_utils import load_keys, shard_keys, selected_rows, atomic_json


def check_lengths(path):
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    pf = pq.ParquetFile(path)
    if not pf.schema_arrow.equals(SCHEMA, check_metadata=False):
        raise ValueError(f'{path}: schema differs')
    scalars = ['n_timesteps', 'n_units', 'map_T', 'map_H', 'map_W']
    total_rows = 0
    for group in range(pf.num_row_groups):
        totals = collections.Counter()
        for batch in pf.iter_batches(row_groups=[group], batch_size=16,
                                      columns=scalars + list(LIST_DTYPES), use_threads=False):
            dims = {}
            for k in scalars:
                if batch.column(k).null_count:
                    raise ValueError(f'{path}: null scalar {k}')
                dims[k] = batch.column(k).to_numpy(zero_copy_only=False).astype(np.int64)
                if np.any(dims[k] <= 0):
                    raise ValueError(f'{path}: nonpositive dimension {k}')
            for column in LIST_DTYPES:
                arr = batch.column(column)
                offsets = arr.offsets.to_numpy(zero_copy_only=False).astype(np.int64)
                if (arr.null_count or arr.values.null_count or offsets[0] < 0 or
                        offsets[-1] > len(arr.values) or np.any(np.diff(offsets) < 0)):
                    raise ValueError(f'{path}: invalid/null offsets {column}')
                expected = np.ones(batch.num_rows, dtype=np.int64)
                shape = DATASETS[column][1] if column in DATASETS else (4 if column == 'playable_area' else 3,)
                for d in shape:
                    expected *= dims[d] if isinstance(d, str) else d
                lengths = np.diff(offsets)
                if not np.array_equal(lengths, expected):
                    raise ValueError(f'{path}: list length {column}, group {group}')
                totals[column] += int(lengths.sum())
            total_rows += batch.num_rows
        # Every list is non-null and non-empty, so num_values equals summed lengths.
        meta = pf.metadata.row_group(group)
        for column in LIST_DTYPES:
            i = pf.schema_arrow.get_field_index(column)
            if totals[column] != meta.column(i).num_values:
                raise ValueError(f'{path}: metadata value count {column}, group {group}')
    return {'shard': str(path), 'rows': total_rows, 'pass': True}


def check_values(task):
    old_shard, selected, new_index, source = task
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    relevant = set(shard_keys(old_shard)) & selected
    by_shard = collections.defaultdict(set)
    for k in relevant:
        by_shard[new_index[k]].add(k)
    new_rows = {}
    for shard, ks in by_shard.items():
        for row in selected_rows(shard, ks):
            new_rows[key(row)] = row
    checked = 0
    for old in selected_rows(old_shard, relevant):
        k = key(old)
        new = new_rows.pop(k)
        for field in ORIGINAL_SCHEMA:
            if not equal_values(old[field.name], new[field.name]):
                raise ValueError(f'{k}: original column differs: {field.name}')
        with h5py.File(Path(source) / relative_path(new), 'r') as f:
            for name in ATTRIBUTE_COLUMNS:
                actual = new[name]
                original = f.attrs[name]
                if name in FRAME_ATTRIBUTES:
                    if np.asarray(original).dtype != np.dtype('int64'):
                        raise ValueError(f'{k}: original attribute dtype {name}')
                    original = int(original)
                if not equal_values(actual, original):
                    raise ValueError(f'{k}: new attribute differs: {name}')
        checked += 1
    if new_rows:
        raise ValueError(f'{old_shard}: unmatched new rows')
    return {'shard': str(old_shard), 'value_rows': checked, 'pass': True}


def scan(root, expected_schema=None):
    rows = collections.defaultdict(list)
    index = {}
    shards = sorted(Path(root).rglob('*.parquet'))
    if not shards:
        raise ValueError(f'{root}: no Parquet shards')
    for path in shards:
        if (expected_schema is not None and
                not pq.ParquetFile(path).schema_arrow.equals(expected_schema, check_metadata=False)):
            raise ValueError(f'{path}: column names, order or types differ')
        keys = shard_keys(path)
        for k in keys:
            if k in index:
                raise ValueError(f'{root}: duplicate key {k}')
            if k[0] not in SPLIT_OUT or path.parent.name != SPLIT_OUT[k[0]]:
                raise ValueError(f'{path}: split directory/value differs for {k}')
            index[k] = str(path)
            rows[k[0]].append(k)
    return rows, index, shards


# Keep datasets and its native libraries out of the process that checks HDF5.
STREAMING_CHECK = r"""
import sys, json, gc, platform
sys.dont_write_bytecode = True
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import datasets
from datasets import load_dataset
plan, cache = json.loads(sys.argv[1]), sys.argv[2]
first = next(iter(plan['data_files'].values()))[0]
schema = pq.ParquetFile(first).schema_arrow
print(schema, flush=True)
half_columns = [f.name for f in schema if pa.types.is_list(f.type) and pa.types.is_float16(f.type.value_type)]
assert len(half_columns) == 9, half_columns
ds = load_dataset('parquet', data_files=plan['data_files'], streaming=True, cache_dir=cache)
results, errors = {}, []
for split, expected in [('train','train'), ('validation','val'), ('test','test')]:
    for column in half_columns:
        if ds[split].features[column].feature.dtype != 'float16':
            errors.append(f'{split}: feature dtype {column}')
    iterator = iter(ds[split])
    count = 0
    # Exhaust every selected file: abandoning a fragment scan can crash at shutdown.
    for row in iterator:
        count += 1
        if row['split'] != expected or set(row) != set(schema.names):
            if len(errors) < 10:
                errors.append(f'{split}: row schema/split at {count}')
        for column in half_columns:
            # Default HF formatting returns Python floats, which have no float16 dtype.
            # Verify the declared type and that this formatting preserves every value.
            raw = np.asarray(row[column])
            values = raw.astype(np.float16)
            if (values.dtype != np.dtype('float16') or
                    not np.array_equal(raw, values.astype(raw.dtype), equal_nan=True)) and len(errors) < 10:
                errors.append(f'{split}: lossy float16 decoding {column}')
            zeros = raw == 0
            if np.any(np.signbit(raw[zeros]) != np.signbit(values[zeros])) and len(errors) < 10:
                errors.append(f'{split}: signed zero changed {column}')
    if count != plan['expected_rows'][split]:
        errors.append(f'{split}: {count} rows != metadata {plan["expected_rows"][split]}')
    results[split] = {'status': 'pass' if not errors else 'fail', 'rows': count,
                      'metadata_rows': plan['expected_rows'][split],
                      'files': plan['data_files'][split], 'float16_columns': len(half_columns),
                      'feature_dtypes': {column: ds[split].features[column].feature.dtype for column in half_columns},
                      'decoded_dtype': 'float16', 'lossless_python_float_formatting': True,
                      'fully_exhausted': True}
    print('STREAMED', split, count, 'rows; float16 columns', len(half_columns), flush=True)
    iterator.close()
    del iterator
    if count:
        del row, values, raw, zeros
    gc.collect()
del ds
gc.collect()
if errors:
    raise ValueError('; '.join(errors))
print('STREAM_RESULT=' + json.dumps({'splits': results, 'schema_printed': True,
      'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                   'pyarrow': pa.__version__, 'datasets': datasets.__version__}}), flush=True)
"""


def check_streaming(args, report):
    if Path(args.streaming_python).absolute() != Path(sys.executable).absolute():
        raise ValueError('Streaming validation must use the current main interpreter')
    cache = args.report.parent / 'hf_cache'
    cache.mkdir(parents=True, exist_ok=True)
    files, expected = {}, {}
    for directory in SPLIT_OUT.values():
        paths = sorted((args.data / directory).glob('*.parquet'))
        paths = paths if args.sample else paths[:1]
        if not paths:
            raise ValueError(f'{directory}: no streaming files')
        files[directory] = [str(p.absolute()) for p in paths]
        expected[directory] = sum(pq.ParquetFile(p).metadata.num_rows for p in paths)
    plan = {'data_files': files, 'expected_rows': expected,
            'selection': 'whole sample splits' if args.sample else 'first sorted shard per split'}
    report['streaming_input'] = plan
    env = dict(os.environ, HF_HOME=str(cache), HF_DATASETS_CACHE=str(cache / 'datasets'),
               HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    child = subprocess.run([sys.executable, '-I', '-u', '-c', STREAMING_CHECK,
                            json.dumps(plan), str(cache / 'datasets')], env=env,
                           capture_output=True, text=True, timeout=1800)
    report['streaming_process'] = {'python': sys.executable, 'exit_code': child.returncode,
                                   'stdout': child.stdout, 'stderr': child.stderr}
    print(child.stdout, end='', flush=True)
    print(child.stderr, end='', file=sys.stderr, flush=True)
    print('HF streaming subprocess exit:', child.returncode, flush=True)
    child.check_returncode()
    result = [line for line in child.stdout.splitlines() if line.startswith('STREAM_RESULT=')]
    if len(result) != 1:
        raise ValueError('Streaming subprocess did not return its summary')
    report['streaming'] = json.loads(result[0].split('=', 1)[1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--old', type=Path, required=True)
    p.add_argument('--src', type=Path, required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--keys', type=Path, help='Fixed full verification sample including edges')
    mode.add_argument('--sample', action='store_true', help='Verify all rows against old sample keys')
    p.add_argument('--workers', type=int, default=32)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--streaming-python', type=Path, default=Path(sys.executable),
                   help='Streaming validation must use the current main interpreter')
    p.add_argument('--skip-streaming', action='store_true',
                   help='Verify items 1-4 only; record item 5 as skipped, never passed')
    args = p.parse_args()
    report = {'pass': False, 'commands': vars(args).copy(), 'length_rows': 0,
              'value_rows': 0, 'counts': {}, 'streaming': {}}
    report['commands'] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    try:
        if not 1 <= args.workers <= 32:
            raise ValueError('--workers must be 1..32')
        rows, index, shards = scan(args.data, SCHEMA)
        old_rows, old_index, old_shards = scan(args.old, ORIGINAL_SCHEMA)
        if args.sample:
            expected = set(old_index)
            selected = expected
        else:
            expected = set()
            for split in SPLIT_OUT:
                lines = (args.src / 'manifests' / f'{split}_manifest.txt').read_text().splitlines()
                ks = [key_from_path(l) for l in lines]
                if len(ks) != len(set(ks)):
                    raise ValueError(f'Duplicate manifest key: {split}')
                expected.update(ks)
            selected = load_keys(args.keys)
        if set(index) != expected or set(index) != set(old_index):
            raise ValueError(f'Key mismatch: missing={len(expected-set(index))}, extra={len(set(index)-expected)}; old/new={len(set(index)^set(old_index))}')
        if not selected <= expected:
            raise ValueError('Requested verification keys missing from data')
        report['counts'] = {s: len(ks) for s, ks in rows.items()}
        print('PASS all keys', report['counts'], flush=True)
        print(SCHEMA, flush=True)
        with get_context('spawn').Pool(args.workers) as pool:
            for result in pool.imap_unordered(check_lengths, map(str, shards)):
                report['length_rows'] += result['rows']
                print('LENGTHS', result['shard'], report['length_rows'], flush=True)
            small_index = {k: index[k] for k in selected}
            tasks = [(str(shard), selected, small_index, str(args.src)) for shard in old_shards
                     if set(shard_keys(shard)) & selected]
            for result in pool.imap_unordered(check_values, tasks):
                report['value_rows'] += result['value_rows']
                print('VALUES', result['shard'], report['value_rows'], flush=True)
        if report['length_rows'] != len(index) or report['value_rows'] != len(selected):
            raise ValueError('Verification row count differs')
        report['data_checks_passed'] = True
        if args.skip_streaming:
            report['streaming'] = {'status': 'skipped', 'reason': '--skip-streaming'}
            report['scope'] = 'items 1-4; item 5 skipped'
            report['complete'] = False
        else:
            check_streaming(args, report)
            report['scope'] = 'items 1-5'
            report['complete'] = True
        report['pass'] = True
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        atomic_json(args.report, report)
        print('PASS' if report['pass'] else 'FAIL', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

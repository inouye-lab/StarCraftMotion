"""Convert all scenario HDF5 files to deterministic, resumable Parquet shards.

Preserves the reviewed columns and appends the missing experiment metadata.
A shard is published by atomic rename only after all its source files pass the
shared schema assertions. No upload functionality is included.
"""
import argparse
import hashlib
import json
import os
import subprocess
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from schema import (SCHEMA, LIST_DTYPES, DATASETS, ATTRIBUTE_COLUMNS, KEY_COLUMNS,
                    SPLIT_OUT, array_shape, assert_source, key_from_path)
from io_utils import atomic_json, shard_keys


def read_hdf5(path):
    path = Path(path)
    row = dict(zip(KEY_COLUMNS, key_from_path(path)))
    with h5py.File(path, 'r') as f:
        t, n, _ = f['unit_data/repeated/coordinate'].shape
        mt, mh, mw = f['map_data/dynamic/creep'].shape
        row.update(n_timesteps=t, n_units=n, map_T=mt, map_H=mh, map_W=mw)
        for k in ATTRIBUTE_COLUMNS:
            v = f.attrs[k]
            row[k] = np.asarray(v, dtype=LIST_DTYPES[k]) if k in LIST_DTYPES else v
        assert_source(f, row, path)
        for k, (p, _) in DATASETS.items():
            row[k] = np.ascontiguousarray(f[p][:], dtype=LIST_DTYPES[k]).ravel()
    return row


def _build_list_array(arrays, np_dtype, value_type):
    lengths = np.array([a.size for a in arrays], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    if offsets[-1] > np.iinfo(np.int32).max:
        raise OverflowError('List offset exceeds int32; reduce row-group size')
    values = np.concatenate(arrays) if arrays else np.empty(0, dtype=np_dtype)
    return pa.ListArray.from_arrays(pa.array(offsets.astype(np.int32)),
                                    pa.array(values, type=value_type))


def rows_to_table(rows):
    arrays = {}
    for field in SCHEMA:
        col = [r[field.name] for r in rows]
        arrays[field.name] = (_build_list_array(col, LIST_DTYPES[field.name],
                                               field.type.value_type)
                              if field.name in LIST_DTYPES else
                              pa.array(col, type=field.type))
    return pa.table(arrays, schema=SCHEMA)


def collect_files(split_dir):
    return sorted(str(p) for p in Path(split_dir).rglob('*.h5'))


def write_shard(task):
    out, paths, row_group_size, fingerprint = task
    out = Path(out)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    expected_keys = [key_from_path(p) for p in paths]
    if out.exists():
        pf = pq.ParquetFile(out)
        if (not pf.schema_arrow.equals(SCHEMA, check_metadata=False) or
                pf.metadata.num_rows != len(paths) or
                (pf.schema_arrow.metadata or {}).get(b'scm_plan') != fingerprint.encode() or
                shard_keys(out) != expected_keys):
            raise ValueError(f'{out}: existing shard is incomplete or belongs to another plan')
        return {'path': str(out), 'rows': len(paths), 'bytes': out.stat().st_size,
                'status': 'skip'}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.parquet.tmp')
    schema = SCHEMA.with_metadata({b'scm_plan': fingerprint.encode()})
    # One task per shard, at most --workers tasks; a worker holds one row group.
    with pq.ParquetWriter(tmp, schema, compression='zstd', compression_level=3) as writer:
        for start in range(0, len(paths), row_group_size):
            rows = [read_hdf5(p) for p in paths[start:start+row_group_size]]
            table = rows_to_table(rows).replace_schema_metadata(schema.metadata)
            writer.write_table(table, row_group_size=row_group_size)
    pf = pq.ParquetFile(tmp)
    if pf.metadata.num_rows != len(paths):
        raise RuntimeError(f'{out}: row count differs from shard plan')
    tmp.replace(out)
    return {'path': str(out), 'rows': len(paths), 'bytes': out.stat().st_size,
            'status': 'written'}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--src', type=Path, required=True)
    p.add_argument('--dst', type=Path, required=True)
    p.add_argument('--splits', nargs='+', choices=list(SPLIT_OUT), default=list(SPLIT_OUT))
    p.add_argument('--workers', type=int, default=32)
    p.add_argument('--files-per-shard', type=int, default=1600)
    p.add_argument('--row-group-size', type=int, default=16)
    return p


def convert(args, files_by_split=None, selection=None):
    if not 1 <= args.workers <= 32:
        raise ValueError('--workers must be between 1 and 32')
    if args.files_per_shard < 1 or args.row_group_size < 1:
        raise ValueError('Shard and row-group sizes must be positive')
    src, dst = args.src.absolute(), args.dst.absolute()
    if dst == src or src in dst.parents or dst in src.parents:
        raise ValueError('Source and destination must be disjoint')
    repo = Path(__file__).resolve().parent.parent
    commit = subprocess.check_output([
        'git', '-C', str(repo), 'log', '-1', '--format=%H', '--',
        'release/convert_to_parquet.py', 'release/sample_convert_to_parquet.py',
        'release/schema.py', 'release/io_utils.py'], text=True).strip()
    files_by_split = files_by_split if files_by_split is not None else {
        s: collect_files(src / s) for s in args.splits}
    for s in args.splits:
        if not files_by_split[s]:
            raise ValueError(f'{src / s}: no files')
    plan = {'source_root': str(src), 'script_git_commit': commit,
            'files_per_shard': args.files_per_shard, 'row_group_size': args.row_group_size,
            'splits': args.splits, 'selection': selection,
            'source_keys_sha256': {s: hashlib.sha256('\n'.join(files_by_split[s]).encode()).hexdigest()
                                   for s in args.splits},
            'counts': {s: len(files_by_split[s]) for s in args.splits}}
    fingerprint = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    dst.mkdir(parents=True, exist_ok=True)
    plan_path = dst / 'conversion_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError(f'{dst}: conversion plan differs; use a new destination')
    atomic_json(plan_path, plan)
    tasks = []
    for split in args.splits:
        files = files_by_split[split]
        for i, start in enumerate(range(0, len(files), args.files_per_shard)):
            tasks.append((str(dst / SPLIT_OUT[split] / f'shard_{i:05d}.parquet'),
                          files[start:start+args.files_per_shard], args.row_group_size, fingerprint))
    expected = {Path(t[0]) for t in tasks}
    if set(dst.rglob('*.parquet')) - expected:
        raise ValueError(f'{dst}: unexpected shards outside conversion plan')
    records = []
    print(json.dumps(plan, indent=2), flush=True)
    with Pool(args.workers) as pool:
        for result in pool.imap_unordered(write_shard, tasks):
            result['path'] = str(Path(result['path']).relative_to(dst))
            records.append(result)
            print(f'{len(records)}/{len(tasks)} {json.dumps(result)}', flush=True)
            atomic_json(dst / 'conversion_manifest.json',
                        {**plan, 'complete': len(records) == len(tasks),
                         'shards': sorted(records, key=lambda r: r['path'])})
    print('PASS conversion complete', flush=True)


if __name__ == '__main__':
    convert(parser().parse_args())

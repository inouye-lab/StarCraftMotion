"""Restore experiment HDF5 files and manifests from camera-ready Parquet.

Standalone dependencies: this file, schema.py, numpy, pyarrow and h5py.
Writes atomically, resumes complete files, and holds one bounded Arrow batch.
Use --keys for a restricted reconstruction, --manifests-only for key-only output.
"""
import argparse
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from schema import (SCHEMA, DATASETS, LIST_DTYPES, KEY_COLUMNS, SPLIT_OUT,
                    key, relative_path, array_shape, rebuild_attributes, equal_values)


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def read_keys(path):
    return [key(r) for r in pq.read_table(path, columns=KEY_COLUMNS,
                                         use_threads=False).to_pylist()]


def selected_rows(path, selected):
    pf = pq.ParquetFile(path)
    if not pf.schema_arrow.equals(SCHEMA, check_metadata=False):
        raise ValueError(f'{path}: incompatible schema')
    keys = read_keys(path)
    offset = 0
    for g in range(pf.num_row_groups):
        n = pf.metadata.row_group(g).num_rows
        if selected is None or any(k in selected for k in keys[offset:offset+n]):
            pos = offset
            for batch in pf.iter_batches(row_groups=[g], batch_size=16, use_threads=False):
                for i in range(batch.num_rows):
                    if selected is not None and keys[pos+i] not in selected:
                        continue
                    row = {}
                    for field, column in zip(batch.schema, batch.columns):
                        v = column[i]
                        row[field.name] = (np.asarray(v.values.to_numpy(zero_copy_only=False),
                                                       dtype=LIST_DTYPES[field.name])
                                           if pa.types.is_list(field.type) else v.as_py())
                    yield row
                pos += batch.num_rows
        offset += n


def populate(f, row):
    """Write typed datasets and exactly the original 17 root attributes."""
    for column, (path, _) in DATASETS.items():
        array = np.asarray(row[column], dtype=LIST_DTYPES[column]).reshape(array_shape(column, row))
        parent, name = path.rsplit('/', 1)
        f.require_group(parent).create_dataset(name, data=array, compression='gzip',
                                               compression_opts=4, shuffle=True)
    for name, value in rebuild_attributes(row).items():
        if isinstance(value, str):
            f.attrs.create(name, value, dtype=h5py.string_dtype('utf-8'))
        else:
            f.attrs.create(name, value)


def complete_file(path, row):
    try:
        with h5py.File(path, 'r') as f:
            attrs = rebuild_attributes(row)
            if set(f.attrs) != set(attrs):
                return False
            for k, v in attrs.items():
                if not equal_values(f.attrs[k], v):
                    return False
                if isinstance(v, str):
                    info = h5py.check_string_dtype(f.attrs.get_id(k).dtype)
                    if info is None or info.encoding != 'utf-8' or info.length is not None:
                        return False
            found = set()
            f.visititems(lambda p, v: found.add(p) if isinstance(v, h5py.Dataset) else None)
            if found != {p for p, _ in DATASETS.values()}:
                return False
            return all(f[p].shape == array_shape(k, row) and
                       f[p].dtype == np.dtype(LIST_DTYPES[k]) for k, (p, _) in DATASETS.items())
    except (OSError, ValueError, KeyError):
        return False


def write_file(out, row):
    path = Path(out) / relative_path(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not complete_file(path, row):
            raise ValueError(f'{path}: existing file is incomplete or incompatible')
        return 'skipped'
    tmp = path.with_suffix('.h5.tmp')
    with h5py.File(tmp, 'w') as f:
        populate(f, row)
    tmp.replace(path)
    return 'written'


def reconstruct_shard(task):
    path, out, selected = task
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    counts = {'written': 0, 'skipped': 0}
    for row in selected_rows(path, selected):
        counts[write_file(out, row)] += 1
    return {'shard': path, **counts}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--src', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--keys', type=Path)
    p.add_argument('--manifests-only', action='store_true')
    p.add_argument('--workers', type=int, default=32)
    p.add_argument('--report', type=Path)
    args = p.parse_args()
    if not 1 <= args.workers <= 32:
        raise ValueError('--workers must be 1..32')
    src, out = args.src.absolute(), args.out.absolute()
    if src == out or src in out.parents or out in src.parents:
        raise ValueError('Source and output must be disjoint')
    selected = {key(r) for r in json.loads(args.keys.read_text())} if args.keys else None
    all_keys, tasks, signatures = set(), [], {}
    manifests = {s: [] for s in SPLIT_OUT}
    shards = sorted(src.rglob('*.parquet'))
    if not shards:
        raise ValueError(f'{src}: no shards')
    for shard in shards:
        ks = read_keys(shard)
        if len(ks) != len(set(ks)) or all_keys.intersection(ks):
            raise ValueError(f'{shard}: duplicate keys')
        all_keys.update(ks)
        chosen = [k for k in ks if selected is None or k in selected]
        for k in chosen:
            manifests[k[0]].append(str(relative_path(dict(zip(KEY_COLUMNS, k)))))
        if chosen:
            tasks.append((str(shard), str(out), set(chosen) if selected is not None else None))
        stat = shard.stat()
        signatures[str(shard.relative_to(src))] = [stat.st_size, stat.st_mtime_ns]
    if selected is not None and not selected <= all_keys:
        raise ValueError(f'Missing requested keys: {len(selected-all_keys)}')
    plan = {'source': str(src), 'shards': signatures,
            'keys': sorted(selected) if selected is not None else None,
            'code_sha256': hashlib.sha256(Path(__file__).read_bytes() +
                                           Path(__file__).with_name('schema.py').read_bytes()).hexdigest()}
    out.mkdir(parents=True, exist_ok=True)
    if not args.manifests_only:
        plan_path = out / 'reconstruction_plan.json'
        if plan_path.exists():
            # JSON converts tuples into lists; normalize before comparing.
            if json.loads(plan_path.read_text()) != json.loads(json.dumps(plan)):
                raise ValueError(f'{out}: reconstruction plan changed')
        elif next(out.rglob('*.h5'), None) is not None:
            raise ValueError(f'{out}: existing HDF5 files without a matching reconstruction plan')
        atomic_json(plan_path, plan)
    (out / 'manifests').mkdir(exist_ok=True)
    for split, lines in manifests.items():
        path = out / 'manifests' / f'{split}_manifest.txt'
        tmp = path.with_suffix('.txt.tmp')
        tmp.write_bytes(''.join(line + '\n' for line in sorted(lines)).encode('utf-8'))
        tmp.replace(path)
    report = {'pass': False, 'source': str(src), 'output': str(out),
              'manifests_only': args.manifests_only,
              'counts': {s: len(v) for s, v in manifests.items()}, 'written': 0, 'skipped': 0}
    report_path = args.report or out / 'reconstruction_manifest.json'
    try:
        if not args.manifests_only:
            with get_context('spawn').Pool(args.workers) as pool:
                for result in pool.imap_unordered(reconstruct_shard, tasks):
                    for k in ['written', 'skipped']:
                        report[k] += result[k]
                    print(json.dumps(result), flush=True)
                    atomic_json(report_path, report)
            if report['written'] + report['skipped'] != sum(report['counts'].values()):
                raise ValueError('Reconstruction row count differs')
        report['pass'] = True
    finally:
        atomic_json(report_path, report)
        print('PASS' if report['pass'] else 'FAIL', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

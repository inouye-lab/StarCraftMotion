"""Compare selected HDF5 files, typed root attributes and exact manifest bytes.

With --keys, filter reference manifest lines to those keys in their original
order. --manifests-only compares all three complete original manifests verbatim.
Never enumerates or reads the reference HDF5 tree outside the selected keys.
"""
import argparse
import json
from multiprocessing import get_context
from pathlib import Path
import h5py
from schema import KEY_COLUMNS, SPLIT_OUT, key, key_from_path, relative_path, equal_values


def inventory(f):
    found = {}
    def visit(path, obj):
        if isinstance(obj, h5py.Dataset):
            found[path] = obj
    f.visititems(visit)
    return found


def compare_open_files(a, b):
    if set(a.attrs) != set(b.attrs):
        return 'root attribute key set'
    for k in a.attrs:
        ai, bi = a.attrs.get_id(k), b.attrs.get_id(k)
        if (ai.dtype != bi.dtype or ai.shape != bi.shape or
                h5py.check_string_dtype(ai.dtype) != h5py.check_string_dtype(bi.dtype)):
            return f'attribute dtype/shape/encoding: {k}'
        if not equal_values(a.attrs[k], b.attrs[k]):
            return f'attribute value/type: {k}'
    aa, bb = inventory(a), inventory(b)
    if set(aa) != set(bb):
        return 'dataset path set'
    for k, av in aa.items():
        bv = bb[k]
        if av.dtype != bv.dtype or av.shape != bv.shape:
            return f'dataset dtype/shape: {k}'
        if not equal_values(av[:], bv[:]):
            return f'dataset values: {k}'
    return None


def compare_file(task):
    ref, other, relative = task
    try:
        with h5py.File(Path(ref) / relative, 'r') as a, h5py.File(Path(other) / relative, 'r') as b:
            difference = compare_open_files(a, b)
        return {'path': relative, 'difference': difference}
    except Exception as e:
        return {'path': relative, 'difference': str(e)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--other', type=Path, required=True)
    p.add_argument('--keys', type=Path)
    p.add_argument('--manifests-only', action='store_true')
    p.add_argument('--workers', type=int, default=32)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    if not 1 <= args.workers <= 32:
        raise ValueError('--workers must be 1..32')
    selected = {key(r) for r in json.loads(args.keys.read_text())} if args.keys else None
    if not args.manifests_only:
        actual = {key_from_path(p) for p in args.other.rglob('*.h5')}
        if selected is None:
            selected = {key_from_path(line) for split in SPLIT_OUT
                        for line in (args.reference / 'manifests' / f'{split}_manifest.txt').read_text().splitlines()}
        if actual != selected:
            raise ValueError(f'HDF5 file key set differs: missing={len(selected-actual)}, extra={len(actual-selected)}')
    report = {'pass': False, 'files_compared': 0, 'manifests': {}, 'differences': [],
              'reference': str(args.reference), 'other': str(args.other),
              'manifest_mode': 'full bytes' if args.manifests_only else 'reference lines filtered by selected keys, original order'}
    try:
        for split in SPLIT_OUT:
            relative = Path('manifests') / f'{split}_manifest.txt'
            expected = (args.reference / relative).read_bytes()
            if selected is not None and not args.manifests_only:
                expected = b''.join(line for line in expected.splitlines(keepends=True)
                                    if key_from_path(line.decode().strip()) in selected)
            actual_bytes = (args.other / relative).read_bytes()
            if actual_bytes != expected:
                report['differences'].append({'path': str(relative), 'difference': 'manifest bytes'})
            else:
                report['manifests'][split] = 'pass'
        if not args.manifests_only:
            paths = [str(relative_path(dict(zip(KEY_COLUMNS, k)))) for k in sorted(selected)]
            tasks = [(str(args.reference), str(args.other), path) for path in paths]
            with get_context('spawn').Pool(args.workers) as pool:
                for result in pool.imap_unordered(compare_file, tasks, chunksize=8):
                    report['files_compared'] += 1
                    if result['difference']:
                        if len(report['differences']) < 20:
                            report['differences'].append(result)
                    if report['files_compared'] % 200 == 0:
                        print('COMPARED', report['files_compared'], flush=True)
        report['pass'] = not report['differences']
    finally:
        tmp = args.report.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(report, indent=2) + '\n')
        tmp.replace(args.report)
        print('PASS' if report['pass'] else 'FAIL', json.dumps(report), flush=True)
    if not report['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

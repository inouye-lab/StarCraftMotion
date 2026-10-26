# Camera-ready Parquet and HDF5 reconstruction

Install numpy, pyarrow and h5py in a separate environment. Verification additionally
uses Hugging Face datasets in a fresh interpreter for a local streaming smoke check.
No command uploads data.

Tested environment: Python 3.13.5, numpy 2.5.4, h5py 3.16.0, pyarrow 26.0.0,
datasets 5.1.0. Streaming validation runs in a fresh isolated subprocess using
that same interpreter and exhausts each selected stream: all sample files, or
the first sorted full shard per split. Row counts must match Parquet metadata;
all nine halffloat columns must arrive as float16 and the child must exit 0.

Abandoning a partially consumed fragment scan of a multi-row-group file can crash
at native shutdown ([apache/arrow#45214](https://github.com/apache/arrow/issues/45214)).
The datasets 5.1 workaround is gated to pyarrow <= 24. Keep row groups at 16.
For a future dataset card: consume a whole shard or peek with
`pq.ParquetFile(path).iter_batches(batch_size=1)`; `next(iter(ds))` followed by exit
may segfault with pyarrow >= 25. Data values and the HDF5 reconstructor are
unaffected: it consumes each selected row group fully with use_threads=False.

`--skip-streaming` remains available and records item 5 as skipped; final release
checks use streaming enabled. All verification workers use spawn. Historical
pinned diagnostic environments are retained but are not used for passing checks.

The 37 reviewed fields retain their names, order, Arrow types and values. The shared
`release/schema.py` appends full xyz start locations as `list<float32>`, playable area
as `list<int64>`, five frame attributes as `int32`, and the unchanged `player_info` and
`replay_info` JSON strings. Every source file must pass the redundancy, dtype, shape,
attribute-key, JSON-key and integer-range assertions. Unexpected metadata aborts
conversion with its source path; it is never silently omitted.

```bash
python release/sample_convert_to_parquet.py --src HDF5_ROOT --dst SAMPLE_OUT \
  --keys-from REVIEWED_SAMPLE_PARQUET --workers 32 --files-per-shard 1600
python release/verify_parquet.py --data SAMPLE_OUT --old REVIEWED_SAMPLE_PARQUET \
  --src HDF5_ROOT --sample --workers 32 \
  --report SAMPLE_VERIFY.json
python release/convert_to_parquet.py --src HDF5_ROOT --dst FULL_OUT \
  --workers 32 --files-per-shard 1600
python release/verify_parquet.py --data FULL_OUT --old REVIEWED_FULL_PARQUET \
  --src HDF5_ROOT --keys VERIFY_KEYS.json --workers 32 \
  --report FULL_VERIFY.json
```

`--splits train val test` selects source splits. The output directories are train,
validation and test; validation rows retain `split="val"`. Input files and shard
boundaries are sorted and deterministic. The sample command uses exactly the reviewed
sample keys. `--seeded-fallback --seed 42` is available when those keys are unavailable.

The full converter distributes shards over at most 32 workers. Each holds one bounded
row group (default 16 scenarios). Int64 accumulation detects list offset overflow
before any int32 conversion. Shards publish atomically. Rerun the exact command to
resume: the saved plan, schema, fingerprint, row counts and all shard keys must match.
The conversion manifest records source root, code commit, split counts and shard rows.
Changing the converter code commit or plan requires a new output directory.
Provenance follows the last commit touching the conversion scripts and shared
modules, so later documentation or round-trip-only commits do not prevent resumption.

The verifier checks every key and every list length. PyArrow has to decompress and
decode Parquet pages, but the all-row check examines only offsets, lengths, null
bitmaps and scalar dimensions. It does not compare list values or read HDF5 for
unselected scenarios. Original float values are compared by bits with NaN equivalence.
The full value sample is a fixed JSON list of key dictionaries, including tagged edges.

## Reconstruct experiment inputs

The reconstructor needs only `parquet_to_hdf5.py`, `schema.py`, numpy, pyarrow and
h5py. It restores `train/`, `val/`, `test/`, map subdirectories and four-digit
segment suffixes, plus lexically sorted UTF-8 manifests ending in LF. Every root
integer attribute is restored as int64, xyz starts as float32, and strings as
variable-length UTF-8. Dataset paths, shapes, dtypes and values are preserved;
physical HDF5 compression is gzip level 4 with shuffle.

```bash
python release/parquet_to_hdf5.py --src SAMPLE_OUT --out HDF5_SAMPLE --workers 32
python release/compare_hdf5.py --reference HDF5_ROOT --other HDF5_SAMPLE \
  --keys SAMPLE_KEYS.json --workers 32 --report SAMPLE_HDF5_COMPARE.json
python release/parquet_to_hdf5.py --src FULL_OUT --out HDF5_SELECTED \
  --keys VERIFY_KEYS.json --workers 32
python release/compare_hdf5.py --reference HDF5_ROOT --other HDF5_SELECTED \
  --keys VERIFY_KEYS.json --workers 32 --report FULL_HDF5_COMPARE.json
python release/parquet_to_hdf5.py --src FULL_OUT --out HDF5_MANIFESTS --manifests-only
python release/compare_hdf5.py --reference HDF5_ROOT --other HDF5_MANIFESTS \
  --manifests-only --report MANIFEST_COMPARE.json
```

`--keys` takes a JSON list of key dictionaries; tagged duplicates are deduplicated.
The task's full reconstruction uses only this fixed verification sample. The
sample reconstruction restores all 1,000 rows. A separate `--manifests-only` run
reads only the four key columns for the full dataset and never decodes array data.

Workers use spawn, at most 32 processes and 16 rows per decoded batch. Files publish
by atomic rename. The reconstruction plan fingerprints code, input shard sizes
and mtimes, and selected keys. Rerun the identical command to resume; complete
files must have compatible dataset metadata and all typed attributes. Existing
incompatible files or a changed plan cause failure instead of overwriting files.

The comparator checks every selected dataset and root attribute, including dtype,
shape, UTF-8 encoding and float bits with NaN equivalence. It checks the complete
output file key set against `--keys`. Subset manifests must equal the original
manifest lines filtered to those keys, preserving original order and bytes. The
separate full-manifest comparison checks all three source files byte for byte.
The reference HDF5 tree is never traversed outside the selected scenario keys.

```bash
python -m unittest discover -s release -p 'test_*.py' -v
```

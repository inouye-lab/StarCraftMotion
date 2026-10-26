"""Bounded Parquet reading and key helpers shared by release commands."""
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from schema import KEY_COLUMNS, LIST_DTYPES, key


def load_keys(path):
    return {key(r) for r in json.loads(Path(path).read_text())}


def shard_keys(path):
    return [key(r) for r in pq.read_table(path, columns=KEY_COLUMNS,
                                        use_threads=False).to_pylist()]


def numpy_row(batch, i):
    row = {}
    for field, column in zip(batch.schema, batch.columns):
        value = column[i]
        row[field.name] = (np.asarray(value.values.to_numpy(zero_copy_only=False),
                                      dtype=LIST_DTYPES[field.name])
                           if pa.types.is_list(field.type) else value.as_py())
    return row


def selected_rows(path, selected=None, columns=None, batch_size=16):
    """Read only row groups that contain requested keys; bound decoded batches."""
    pf = pq.ParquetFile(path)
    keys = shard_keys(path)
    offset = 0
    for g in range(pf.num_row_groups):
        n = pf.metadata.row_group(g).num_rows
        if selected is None or any(k in selected for k in keys[offset:offset+n]):
            pos = offset
            for batch in pf.iter_batches(row_groups=[g], columns=columns,
                                          batch_size=batch_size, use_threads=False):
                for i in range(batch.num_rows):
                    if selected is None or keys[pos+i] in selected:
                        yield numpy_row(batch, i)
                pos += batch.num_rows
        offset += n


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)

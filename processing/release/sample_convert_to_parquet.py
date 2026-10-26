"""Create the reviewer sample using its existing keys, with a seeded fallback."""
import random
from collections import defaultdict
from pathlib import Path
from convert_to_parquet import collect_files, convert, parser
from io_utils import shard_keys
from schema import SPLIT_OUT, relative_path, KEY_COLUMNS

def sample_with_map_coverage(by_map: dict, n: int, rng: random.Random) -> list:
    """Sample n files ensuring at least one file per map, then fill remaining slots
    proportionally to each map's size."""
    maps = sorted(by_map.keys())
    n_maps = len(maps)
    assert n >= n_maps, f"Cannot cover {n_maps} maps with only {n} samples"

    # guarantee one per map first
    selected = [rng.choice(by_map[m]) for m in maps]
    remaining_pool = {m: [f for f in by_map[m] if f not in selected] for m in maps}
    total_remaining = sum(len(v) for v in remaining_pool.values())

    # fill remaining slots proportionally
    slots_left = n - n_maps
    extra = []
    for m in maps:
        quota = round(slots_left * len(remaining_pool[m]) / max(total_remaining, 1))
        extra.extend(rng.sample(remaining_pool[m], min(quota, len(remaining_pool[m]))))

    # top up or trim to exactly n
    all_extra = [f for m in maps for f in remaining_pool[m] if f not in extra]
    while len(selected) + len(extra) < n and all_extra:
        extra.append(all_extra.pop(rng.randrange(len(all_extra))))
    selected.extend(extra[:n - len(selected)])

    rng.shuffle(selected)
    return selected



def main():
    p = parser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--keys-from', type=Path,
                       help='Reviewed sample Parquet root; preserves its exact scenarios')
    group.add_argument('--seeded-fallback', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    files = {s: [] for s in args.splits}
    if args.keys_from:
        keys = [k for shard in sorted(args.keys_from.rglob('*.parquet')) for k in shard_keys(shard)]
        if len(keys) != len(set(keys)):
            raise ValueError('Duplicate reviewed sample keys')
        for k in keys:
            if k[0] in files:
                files[k[0]].append(str(args.src.absolute() / relative_path(dict(zip(KEY_COLUMNS, k)))))
        selection = {'keys_from': str(args.keys_from.absolute())}
    else:
        rng = random.Random(args.seed)
        for split in args.splits:
            by_map = defaultdict(list)
            for f in collect_files(args.src / split):
                by_map[Path(f).parent.name].append(f)
            files[split] = sample_with_map_coverage(by_map, 800 if split == 'train' else 100, rng)
        selection = {'seeded_fallback': args.seed}
    files = {s: sorted(fs) for s, fs in files.items()}
    convert(args, files, selection)


if __name__ == '__main__':
    main()

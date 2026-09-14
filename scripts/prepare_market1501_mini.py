"""Create a small Market1501 subset in the layout expected by FedCompass."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a small Market1501 subset for FedCompass ReID training.")
    parser.add_argument("--source", required=True, help="Path to extracted Market-1501-v15.09.15")
    parser.add_argument("--output", default="real_data/market1501_mini", help="Output directory for the mini subset")
    parser.add_argument("--train-identities", type=int, default=30, help="Number of identities to keep for train")
    parser.add_argument("--query-per-identity", type=int, default=1, help="Number of query images per identity")
    parser.add_argument("--gallery-per-identity", type=int, default=2, help="Number of gallery images per identity")
    parser.add_argument("--train-per-identity", type=int, default=4, help="Number of train images per identity")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    train_src = source_root / "bounding_box_train"
    query_src = source_root / "query"
    gallery_src = source_root / "bounding_box_test"
    if not train_src.exists():
        raise FileNotFoundError(f"Could not find {train_src}")

    if output_root.exists():
        shutil.rmtree(output_root)
    for split_name in ("train", "query", "gallery"):
        (output_root / split_name).mkdir(parents=True, exist_ok=True)

    identities = sorted(
        {
            path.name.split("_")[0]
            for path in train_src.glob("*.jpg")
            if path.name[:4].isdigit() and not path.name.startswith("-1")
        }
    )[: args.train_identities]
    for identity in identities:
        train_count = _copy_subset(train_src, output_root / "train" / identity, identity, args.train_per_identity)
        query_count = _copy_subset(query_src, output_root / "query" / identity, identity, args.query_per_identity)
        gallery_count = _copy_subset(gallery_src, output_root / "gallery" / identity, identity, args.gallery_per_identity)
        if query_count == 0 or gallery_count == 0:
            _synthesize_eval_from_train(
                train_src=train_src,
                query_target=output_root / "query" / identity,
                gallery_target=output_root / "gallery" / identity,
                identity=identity,
                train_per_identity=args.train_per_identity,
                query_per_identity=args.query_per_identity,
                gallery_per_identity=args.gallery_per_identity,
            )

    print(f"Prepared mini Market1501 subset at {output_root}")


def _copy_subset(source_dir: Path, target_dir: Path, identity: str, limit: int) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    matches = sorted(path for path in source_dir.glob(f"{identity}_*.jpg"))
    for path in matches[:limit]:
        shutil.copy2(path, target_dir / path.name)
    return min(len(matches), limit)


def _synthesize_eval_from_train(
    *,
    train_src: Path,
    query_target: Path,
    gallery_target: Path,
    identity: str,
    train_per_identity: int,
    query_per_identity: int,
    gallery_per_identity: int,
) -> None:
    matches = sorted(path for path in train_src.glob(f"{identity}_*.jpg"))
    if not matches:
        return
    query_target.mkdir(parents=True, exist_ok=True)
    gallery_target.mkdir(parents=True, exist_ok=True)
    query_candidates = matches[:query_per_identity]
    gallery_start = min(train_per_identity, len(matches))
    gallery_candidates = matches[gallery_start:gallery_start + gallery_per_identity]
    if not gallery_candidates:
        gallery_candidates = matches[query_per_identity:query_per_identity + gallery_per_identity]
    for path in query_candidates:
        shutil.copy2(path, query_target / path.name)
    for path in gallery_candidates:
        shutil.copy2(path, gallery_target / path.name)


if __name__ == "__main__":
    main()

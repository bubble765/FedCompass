#!/usr/bin/env python3
"""Download ModelNet40 parquet shards temporarily, build a compact mini dataset, then delete the full shards."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


HF_DATASET_BASE = "https://huggingface.co/datasets/jxie/modelnet40/resolve/main/data"
TRAIN_SHARDS = [
    "train-00000-of-00003.parquet",
    "train-00001-of-00003.parquet",
    "train-00002-of-00003.parquet",
]
TEST_SHARDS = [
    "test-00000-of-00001.parquet",
]

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
REAL_DATA_ROOT = WORKSPACE_ROOT / "real_data"
OUTPUT_DIR = REAL_DATA_ROOT / "modelnet40_mini"
OUTPUT_FILE = OUTPUT_DIR / "modelnet40_mini.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", type=int, default=40, help="Number of classes to keep.")
    parser.add_argument("--train-per-class", type=int, default=60, help="Train samples kept per class.")
    parser.add_argument("--test-per-class", type=int, default=20, help="Test samples kept per class.")
    parser.add_argument("--points-per-shape", type=int, default=1024, help="Number of points kept per shape.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--local-train-shards", nargs="*", default=None, help="Optional local parquet train shard paths.")
    parser.add_argument("--local-test-shards", nargs="*", default=None, help="Optional local parquet test shard paths.")
    return parser.parse_args()


def download_shards(target_dir: Path, shard_names: list[str]) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for shard_name in shard_names:
        shard_path = target_dir / shard_name
        subprocess.run(["curl", "-L", f"{HF_DATASET_BASE}/{shard_name}", "-o", str(shard_path)], check=True)
        downloaded.append(shard_path)
    return downloaded


def load_dataframe(shard_paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in shard_paths]
    return pd.concat(frames, ignore_index=True)


def normalize_points(value, point_count: int, rng: np.random.Generator) -> np.ndarray:
    if hasattr(value, "tolist"):
        value = value.tolist()
    points = np.asarray(value)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Unexpected point cloud shape: {points.shape}")
    points = points[:, :3].astype(np.float32, copy=False)
    if len(points) >= point_count:
        indices = rng.choice(len(points), size=point_count, replace=False)
    else:
        indices = rng.choice(len(points), size=point_count, replace=True)
    sampled = points[indices]
    sampled -= sampled.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(sampled, axis=1).max()
    if scale > 0:
        sampled /= scale
    return sampled.astype(np.float32)


def build_split(
    frame: pd.DataFrame,
    selected_labels: list[int],
    per_class: int,
    point_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    point_sets: list[np.ndarray] = []
    labels: list[int] = []
    label_remap = {label: idx for idx, label in enumerate(selected_labels)}

    for raw_label in selected_labels:
        class_rows = frame[frame["label"] == raw_label]
        if class_rows.empty:
            continue
        if len(class_rows) <= per_class:
            sampled_rows = class_rows
        else:
            sampled_rows = class_rows.sample(n=per_class, random_state=seed + raw_label)
        for _, row in sampled_rows.iterrows():
            point_sets.append(normalize_points(row["inputs"], point_count, rng))
            labels.append(label_remap[raw_label])

    if not point_sets:
        raise RuntimeError("No samples collected for the requested split.")
    return np.stack(point_sets, axis=0), np.asarray(labels, dtype=np.int64)


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.local_train_shards and args.local_test_shards:
        train_shards = [Path(path) for path in args.local_train_shards]
        test_shards = [Path(path) for path in args.local_test_shards]
    else:
        with tempfile.TemporaryDirectory(prefix="modelnet40_hf_") as tmpdir:
            tmp_path = Path(tmpdir)
            train_shards = download_shards(tmp_path / "train", TRAIN_SHARDS)
            test_shards = download_shards(tmp_path / "test", TEST_SHARDS)

            train_df = load_dataframe(train_shards)
            test_df = load_dataframe(test_shards)
            selected_labels = sorted(int(label) for label in train_df["label"].drop_duplicates().tolist())[: args.classes]
            if len(selected_labels) < args.classes:
                raise RuntimeError(f"Expected at least {args.classes} labels, found {len(selected_labels)}")

            train_points, train_labels = build_split(
                train_df,
                selected_labels,
                args.train_per_class,
                args.points_per_shape,
                args.seed,
            )
            test_points, test_labels = build_split(
                test_df,
                selected_labels,
                args.test_per_class,
                args.points_per_shape,
                args.seed + 1,
            )

            np.savez_compressed(
                OUTPUT_FILE,
                train_points=train_points,
                train_labels=train_labels,
                test_points=test_points,
                test_labels=test_labels,
                class_names=np.asarray([f"class_{label}" for label in selected_labels], dtype=object),
                source_url="https://huggingface.co/datasets/jxie/modelnet40",
            )

            # Explicit cleanup before tempdir removal, so no full parquet shards remain.
            shutil.rmtree(tmp_path, ignore_errors=True)
    if args.local_train_shards and args.local_test_shards:
        train_df = load_dataframe(train_shards)
        test_df = load_dataframe(test_shards)
        selected_labels = sorted(int(label) for label in train_df["label"].drop_duplicates().tolist())[: args.classes]
        if len(selected_labels) < args.classes:
            raise RuntimeError(f"Expected at least {args.classes} labels, found {len(selected_labels)}")

        train_points, train_labels = build_split(
            train_df,
            selected_labels,
            args.train_per_class,
            args.points_per_shape,
            args.seed,
        )
        test_points, test_labels = build_split(
            test_df,
            selected_labels,
            args.test_per_class,
            args.points_per_shape,
            args.seed + 1,
        )

        np.savez_compressed(
            OUTPUT_FILE,
            train_points=train_points,
            train_labels=train_labels,
            test_points=test_points,
            test_labels=test_labels,
            class_names=np.asarray([f"class_{label}" for label in selected_labels], dtype=object),
            source_url="local_parquet",
        )

    for path in OUTPUT_DIR.iterdir():
        if path != OUTPUT_FILE and path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path != OUTPUT_FILE and path.is_file():
            path.unlink(missing_ok=True)

    print(f"Saved mini dataset to {OUTPUT_FILE}")
    print(f"train shape: {train_points.shape}, test shape: {test_points.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

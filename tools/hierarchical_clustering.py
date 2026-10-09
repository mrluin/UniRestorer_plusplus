import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "feature_dir": "outputs/dr_features",
    "output_dir": "outputs/hierarchical_clustering",
    "depth": 2,  # Non-root levels: root / two groups / four leaves.
    "normalization": "l2",  # Normalize raw GAP vectors, not cluster centers.
    "seed": 19,
    "n_init": 10,
    "max_iter": 100,
    "tol": 1.0e-6,
    "batch_size": 16384,
}


def _chunks(features, indices, batch_size, normalization):
    for start in range(0, len(indices), batch_size):
        points = np.asarray(features[indices[start:start + batch_size]], dtype=np.float64)
        if normalization == "l2":
            scale = np.abs(points).max(axis=1, keepdims=True)
            if np.any(scale == 0):
                raise ValueError("L2 normalization is undefined for a zero feature vector")
            points = points / scale
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        yield start, points


def _assign(features, indices, centers, batch_size, normalization):
    labels = np.empty(len(indices), dtype=np.int64)
    distances = np.empty(len(indices), dtype=np.float64)
    for start, points in _chunks(features, indices, batch_size, normalization):
        squared = np.stack([np.square(points - center).sum(1) for center in centers], axis=1)
        chosen = squared.argmin(1)
        stop = start + len(points)
        labels[start:stop] = chosen
        distances[start:stop] = squared[np.arange(len(points)), chosen]
    return labels, distances


def _means(features, indices, labels, groups, batch_size, normalization):
    sums = np.zeros((groups, features.shape[1]), dtype=np.float64)
    counts = np.bincount(labels, minlength=groups)
    if np.any(counts == 0):
        raise ValueError("Lloyd produced an empty child cluster")
    for start, points in _chunks(features, indices, batch_size, normalization):
        local = labels[start:start + len(points)]
        for group in range(groups):
            sums[group] += points[local == group].sum(0)
    means = sums / counts[:, None]
    if not np.isfinite(means).all():
        raise ValueError("Feature mean accumulation exceeded the float64 range")
    return means


def _two_means(features, indices, rng, n_init, max_iter, tol, batch_size, normalization):
    if len(indices) < 2:
        raise ValueError("A parent needs at least two samples to split")
    best = None
    failures = []
    for initialization in range(n_init):
        first = int(rng.integers(len(indices)))
        center = next(_chunks(features, indices[first:first + 1], batch_size, normalization))[1][0]
        _, weights = _assign(features, indices, center[None], batch_size, normalization)
        if weights.sum() == 0:
            raise ValueError("A parent needs at least two distinct vectors after normalization")
        second = int(rng.choice(len(indices), p=weights / weights.sum()))
        other = next(_chunks(features, indices[second:second + 1], batch_size, normalization))[1][0]
        centers = np.stack([center, other])
        previous = None
        for iteration in range(1, max_iter + 1):
            labels, _ = _assign(features, indices, centers, batch_size, normalization)
            try:
                updated = _means(features, indices, labels, 2, batch_size, normalization)
            except ValueError as error:
                failures.append(str(error))
                break
            movement = np.linalg.norm(updated - centers, axis=1).max()
            # A tolerance/cap alone cannot justify inconsistent labels/means.
            if movement <= tol or np.array_equal(labels, previous) or iteration == max_iter:
                final_labels, distances = _assign(features, indices, updated, batch_size, normalization)
                if np.array_equal(labels, final_labels):
                    inertia = float(distances.sum())
                    if best is None or inertia < best[0]:
                        best = (inertia, labels, updated, initialization, iteration)
                    break
            centers, previous = updated, labels
        else:
            failures.append("Lloyd did not converge within max_iter")
    if best is None:
        raise ValueError("No initialization converged; increase max_iter or inspect the features. " +
                         "; ".join(sorted(set(failures))))
    inertia, labels, centers, initialization, iterations = best
    info = dict(inertia=inertia, chosen_initialization=initialization, iterations=iterations,
                failed_initializations=len(failures), failure_reasons=sorted(set(failures)))
    return labels, centers, info


def cluster_features(options):
    """Write one hierarchy artifact from resolved options; return its summary."""
    normalization = options["normalization"]
    if normalization not in ("none", "l2"):
        raise ValueError("Set normalization explicitly to 'none' or 'l2'; the paper does not choose it")
    feature_dir = Path(options["feature_dir"])
    output_dir = Path(options["output_dir"])
    if output_dir.exists():
        raise FileExistsError("Clustering output directory already exists: " + str(output_dir))
    features = np.load(feature_dir / "features.npy", mmap_mode="r", allow_pickle=False)
    try:
        count, p = features.shape
        if p == 0:
            raise ValueError("features.npy must contain a nonempty N-by-p per-image feature matrix")
        depth = options["depth"]
        if count < 2 ** depth:
            raise ValueError("A full depth-D tree requires at least 2**D samples")
        samples = feature_dir / "samples.csv"
        with samples.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = 0
            sample_ids = set()
            for rows, row in enumerate(csv.DictReader(stream), 1):
                if int(row["row"]) != rows - 1:
                    raise ValueError("samples.csv row must follow features.npy order 0..N-1")
                sample_id = row["sample_id"].strip()
                if not sample_id or sample_id in sample_ids:
                    raise ValueError("samples.csv requires nonempty unique sample_id values")
                sample_ids.add(sample_id)
        if rows != count:
            raise ValueError("samples.csv row count does not match features.npy")
        source_metadata = feature_dir / "metadata.json"
        source_info = json.loads(source_metadata.read_text(encoding="utf-8"))
        if source_info.get("completed") is not True:
            raise ValueError("The feature extraction metadata does not mark a completed artifact")

        seed = options.get("seed", 19)
        n_init = options.get("n_init", 10)
        max_iter = options.get("max_iter", 100)
        tol = options.get("tol", 1e-6)
        batch_size = options.get("batch_size", 16384)
        rng = np.random.default_rng(seed)
        levels = depth + 1
        node_count = 2 ** levels - 1
        labels = np.zeros((count, levels), dtype=np.int64)
        centers = np.empty((node_count, p), dtype=np.float64)
        raw_centers = np.empty_like(centers) if normalization == "l2" else centers
        groups = [np.arange(count, dtype=np.int64)]
        centers[0] = _means(features, groups[0], labels[:, 0], 1, batch_size, normalization)[0]
        nodes = []
        for level in range(levels):
            following = []
            for group, indices in enumerate(groups):
                heap_id = 2 ** level - 1 + group
                node = dict(heap_id=heap_id, level=level, group=group, center_row=heap_id,
                            parent=-1 if heap_id == 0 else (heap_id - 1) // 2,
                            children=[] if level == depth else [2 * heap_id + 1, 2 * heap_id + 2],
                            expert_id=heap_id - 1, sample_count=len(indices))
                nodes.append(node)
                if normalization == "l2":
                    raw_centers[heap_id] = _means(features, indices, np.zeros(len(indices), dtype=np.int64),
                                                  1, batch_size, "none")[0]
                if level == depth:
                    continue
                try:
                    assigned, child_centers, info = _two_means(
                        features, indices, rng, n_init, max_iter, tol, batch_size, normalization)
                except ValueError as error:
                    raise ValueError("Cannot split heap node {} (level {}, {} samples): {}".format(
                        heap_id, level, len(indices), error)) from error
                node["split"] = info
                for child in (0, 1):
                    child_indices = indices[assigned == child]
                    child_group = 2 * group + child
                    labels[child_indices, level + 1] = child_group
                    centers[2 * heap_id + 1 + child] = child_centers[child]
                    following.append(child_indices)
            groups = following

        level_sizes = [2 ** level for level in range(levels)]
        parents = [[]] + [[group // 2 for group in range(size)] for size in level_sizes[1:]]
        leaves = 2 ** depth
        ancestors = [[leaf // 2 ** (depth - level) for leaf in range(leaves)] for level in range(levels)]
        hierarchy = dict(schema_version=1, artifact_type="recursive_parent_kmeans", completed=True,
                         N=count, p=p, depth=depth, L=levels, level_sizes=level_sizes,
                         parents_by_level=parents, leaf_to_group=ancestors,
                         expert_offsets=[0] + [2 ** level - 2 for level in range(1, levels)],
                         normalization=normalization,
                         center_space="l2_normalized_dr_gap" if normalization == "l2" else "raw_dr_gap",
                         raw_center_space="raw_dr_gap",
                         raw_centers_purpose="Original GAP member means for a future prior/NLL coordinate comparison",
                         kmeans=dict(distance="squared_euclidean", initialization="kmeans++", branching=2,
                                     seed=seed, n_init=n_init, max_iter=max_iter, tol=tol, batch_size=batch_size),
                         files=dict(labels="labels.npy", centers="centers.npy", samples="samples.csv"),
                         source_metadata=source_info, nodes=nodes)
        # All splits succeeded before an output directory is created. The completed
        # hierarchy is written last, so a write failure cannot publish success.
        output_dir.mkdir(parents=True, exist_ok=False)
        np.save(output_dir / "labels.npy", labels, allow_pickle=False)
        np.save(output_dir / "centers.npy", centers, allow_pickle=False)
        np.save(output_dir / "raw_centers.npy", raw_centers, allow_pickle=False)
        hierarchy["files"]["raw_centers"] = "raw_centers.npy"
        shutil.copyfile(samples, output_dir / "samples.csv")
        shutil.copyfile(source_metadata, output_dir / "source_metadata.json")
        hierarchy["files"]["source_metadata"] = "source_metadata.json"
        (output_dir / "hierarchy.json").write_text(json.dumps(hierarchy, indent=2) + "\n", encoding="utf-8")
        return dict(output_dir=str(output_dir), n_samples=count, p=p, depth=depth, L=levels,
                    num_nodes=node_count, normalization=normalization, files=hierarchy["files"])
    finally:
        features._mmap.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run hierarchical clustering using CONFIG in this file.")
    parser.parse_args(argv)
    options = dict(CONFIG, feature_dir=ROOT / CONFIG["feature_dir"],
                   output_dir=ROOT / CONFIG["output_dir"])
    print(json.dumps(cluster_features(options), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

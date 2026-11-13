#!/usr/bin/env python3
"""Measure duplicate-detection recall / precision against ``ground_truth.json``.

    python scripts/eval_dedup.py --dataset data/chaos --data-dir data/demo_index

``--data-dir`` must hold an index of ``--dataset`` (e.g. produced by
``scripts/demo.py``); pass ``--index`` to build one first (int8, OCR off).
A planted pair counts as recalled when both files land in the same group.
Pair precision counts every within-group pair as a true positive when both
files belong to the same planted duplicate family (source + its copies).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("VS_MODELS_DIR", str(ROOT / "data" / "models"))


def evaluate(payload: dict, ground_truth: dict, dataset: Path) -> dict:
    """Score a ``/duplicates`` payload against the planted pairs."""
    root = dataset.resolve()

    def rel(p: str) -> str:
        try:
            return str(Path(p).resolve().relative_to(root))
        except ValueError:
            return p

    group_of: dict[str, int] = {}
    for g in payload["groups"]:
        for im in g["images"]:
            group_of[rel(im["file_path"])] = g["group_id"]

    # duplicate families = connected components of the planted pairs
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = ground_truth["duplicate_pairs"]
    for pr in pairs:
        parent[find(pr["a"])] = find(pr["b"])

    by_kind: dict[str, list[int]] = {}
    by_transform: dict[str, list[int]] = {}
    misses = []
    for pr in pairs:
        ga, gb = group_of.get(pr["a"]), group_of.get(pr["b"])
        hit = int(ga is not None and ga == gb)
        by_kind.setdefault(pr["kind"], []).append(hit)
        if pr["kind"] == "near":
            by_transform.setdefault(pr.get("transform", "?"), []).append(hit)
        if not hit:
            misses.append(pr)

    tp = fp = 0
    fp_examples = []
    for g in payload["groups"]:
        files = [rel(im["file_path"]) for im in g["images"]]
        for a, b in combinations(files, 2):
            if a in parent and b in parent and find(a) == find(b):
                tp += 1
            else:
                fp += 1
                if len(fp_examples) < 8:
                    cat = ground_truth["files"]
                    fp_examples.append((a, cat.get(a, {}).get("category"), b, cat.get(b, {}).get("category")))
    all_hits = [h for v in by_kind.values() for h in v]
    sizes = sorted((len(g["images"]) for g in payload["groups"]), reverse=True)
    return {
        "planted_pairs": len(pairs),
        "recall": round(sum(all_hits) / max(1, len(all_hits)), 4),
        "recall_by_kind": {k: round(sum(v) / len(v), 4) for k, v in by_kind.items()},
        "recall_by_transform": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_transform.items())},
        "pair_precision": round(tp / max(1, tp + fp), 4),
        "within_group_pairs": tp + fp,
        "n_groups": payload["n_groups"],
        "n_images": payload["n_images"],
        "largest_groups": sizes[:5],
        "misses": misses[:10],
        "false_positive_examples": fp_examples,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=ROOT / "data" / "chaos")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "demo_index")
    ap.add_argument("--index", action="store_true", help="index --dataset into --data-dir first")
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--max-hamming", type=int, default=6)
    ap.add_argument("--sweep", action="store_true", help="also try a few threshold / hamming settings")
    a = ap.parse_args(argv)

    os.environ["SEARCH_DATA_DIR"] = str(a.data_dir.resolve())
    os.environ.setdefault("VS_EMBED_BACKEND", "onnx")
    os.environ.setdefault("VS_ONNX_PRECISION", "int8")
    from app.db.schema import get_db, init_db
    from app.dedup import DedupParams, find_duplicates

    init_db()
    if a.index:
        os.environ.setdefault("VS_OCR_ENGINE", "none")
        from app.indexer.worker import IndexWorker
        path = str(a.dataset.resolve())
        with get_db() as conn:
            row = conn.execute("SELECT folder_id FROM watched_folders WHERE path=?", (path,)).fetchone()
            fid = row[0] if row else conn.execute("INSERT INTO watched_folders (path) VALUES (?)", (path,)).lastrowid
        t0 = time.time()
        asyncio.run(IndexWorker().start([fid]))
        print(f"indexed in {time.time() - t0:.1f}s", file=sys.stderr)

    gt = json.loads((a.dataset / "ground_truth.json").read_text())
    settings = [(a.threshold, a.max_hamming)]
    if a.sweep:
        settings += [(0.97, 6), (0.93, 6), (0.95, 4), (0.95, 8), (1.01, 6), (0.95, -1)]
    out = []
    for thr, ham in settings:
        t0 = time.perf_counter()
        payload = find_duplicates(DedupParams(threshold=thr, max_hamming=ham))
        ms = (time.perf_counter() - t0) * 1000
        r = evaluate(payload, gt, a.dataset)
        r.update({"threshold": thr, "max_hamming": ham, "dedup_ms": round(ms, 1)})
        out.append(r)
        print(json.dumps({k: v for k, v in r.items() if k not in ("misses",)}, indent=None))
    if len(out) == 1:
        print(json.dumps(out[0]["misses"], indent=1))


if __name__ == "__main__":
    main()

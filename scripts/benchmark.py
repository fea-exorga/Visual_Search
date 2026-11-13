#!/usr/bin/env python3
"""Benchmark encoders, the indexing pipeline and FAISS; write docs/BENCHMARKS.md.

    python scripts/benchmark.py --dataset data/chaos --n 400

Sections
  1. Encoder (fp32 / fp16 / int8): image throughput at batch 32 on pre-decoded
     images, text-query latency, embedding agreement vs fp32 (mean cosine) and
     retrieval agreement (top-10 overlap with fp32 on 20 text queries).
  2. Full indexing throughput with ``IndexWorker`` (scan → decode → colour →
     OCR → hashes → embed → DB) on a throw-away ``SEARCH_DATA_DIR``, OCR on/off,
     plus the pre-M5 sequential worker (from git HEAD) as a baseline.
  3. FAISS top-k query latency p50 / p99 (``app.search.faiss_index`` when
     available, raw FAISS otherwise).
  4. Duplicate-detection recall / precision on a full index of the dataset
     (``--dedup-index``, built by ``scripts/demo.py``) vs ``ground_truth.json``.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
os.environ.setdefault("VS_MODELS_DIR", str(ROOT / "data" / "models"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

QUERIES = [
    "a lecture slide about gradient descent", "source code in a dark editor", "a funny meme with a caption",
    "sunset over the mountains", "a city skyline at night", "a pine forest", "a bar chart", "a line chart",
    "a hand drawn sketch of a house", "a colourful abstract palette", "a mobile app user interface",
    "the ocean at dusk", "a smiley face", "a stick figure drawing", "neural networks lecture",
    "a settings screen", "stars in the night sky", "a green landscape", "a white slide with bullet points",
    "a terminal with python code",
]
PRECISIONS = ["fp32", "fp16", "int8"]
BASELINE_REV = "e30029a"  # foundation commit: last pre-M5 worker


def log(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


def machine_info() -> dict:
    import onnxruntime as ort

    cpu = platform.processor() or ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        import faiss
        faiss_v = faiss.__version__
    except Exception:
        faiss_v = "n/a"
    try:
        tess = subprocess.run(["tesseract", "--version"], capture_output=True, text=True).stdout.split("\n")[0]
    except Exception:
        tess = "n/a"
    mem = ""
    try:
        kb = int(next(l for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemTotal")).split()[1])
        mem = f"{kb / 1024 / 1024:.1f} GiB"
    except Exception:
        pass
    try:
        load = " / ".join(f"{x:.2f}" for x in os.getloadavg())
    except OSError:
        load = "n/a"
    return {
        "load_avg_at_start": load,
        "cpu": cpu, "cpu_count": os.cpu_count(), "memory": mem, "platform": platform.platform(),
        "python": platform.python_version(), "onnxruntime": ort.__version__,
        "providers": ",".join(ort.get_available_providers()), "numpy": np.__version__,
        "faiss": faiss_v, "pillow": Image.__version__, "tesseract": tess,
    }


def list_images(dataset: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return sorted(p for p in dataset.rglob("*") if p.suffix.lower() in exts and not p.name.startswith("."))


def pct(xs: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(xs), q)) if xs else float("nan")


# ── 1. encoders ───────────────────────────────────────────────────────


def bench_encoders(files: list[Path], batch: int) -> dict:
    from app.embeddings.onnx_clip import OnnxClipEncoder
    from app.indexer.worker import WORK_MAX_SIDE, load_image

    log(f"[enc] decoding {len(files)} images …")
    imgs = [load_image(p, WORK_MAX_SIDE) for p in files]
    out: dict = {"n": len(imgs), "batch": batch, "rows": {}}
    embs: dict[str, np.ndarray] = {}
    tembs: dict[str, np.ndarray] = {}
    for prec in PRECISIONS:
        t0 = time.perf_counter()
        enc = OnnxClipEncoder(precision=prec, model_path=Path(os.environ["VS_MODELS_DIR"]) / "Xenova__clip-vit-base-patch32")
        load_s = time.perf_counter() - t0
        enc.encode_images(imgs[:4])  # warm-up
        enc.encode_texts(["warm up"])
        # preprocessing alone
        t0 = time.perf_counter()
        pre = [enc.preprocess_uint8(im) for im in imgs]
        pre_s = time.perf_counter() - t0
        # model only (pre-processed batches)
        t0 = time.perf_counter()
        vecs = []
        for i in range(0, len(pre), batch):
            vecs.append(enc.encode_pixel_batch(enc.normalize_batch(np.stack(pre[i:i + batch]))))
        model_s = time.perf_counter() - t0
        # end-to-end encode_images (preprocess + model)
        t0 = time.perf_counter()
        for i in range(0, len(imgs), batch):
            enc.encode_images(imgs[i:i + batch])
        e2e_s = time.perf_counter() - t0
        embs[prec] = np.concatenate(vecs)
        lat = []
        for q in QUERIES * 2:
            t0 = time.perf_counter()
            enc.encode_texts([q])
            lat.append((time.perf_counter() - t0) * 1000)
        tembs[prec] = enc.encode_texts(QUERIES)
        out["rows"][prec] = {
            "load_s": round(load_s, 2),
            "img_per_s_model": round(len(imgs) / model_s, 1),
            "img_per_s_e2e": round(len(imgs) / e2e_s, 1),
            "preprocess_ms_per_img": round(pre_s / len(imgs) * 1000, 2),
            "text_ms_p50": round(pct(lat, 50), 1),
            "text_ms_p99": round(pct(lat, 99), 1),
        }
        log(f"[enc] {prec}: {out['rows'][prec]}")
        del enc
    ref, tref = embs["fp32"], tembs["fp32"]
    ref_top = np.argsort(-(tref @ ref.T), axis=1)[:, :10]
    for prec in PRECISIONS:
        cos = np.einsum("ij,ij->i", embs[prec], ref)
        top = np.argsort(-(tembs[prec] @ embs[prec].T), axis=1)[:, :10]
        overlap = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(top, ref_top)])
        out["rows"][prec].update({
            "cos_vs_fp32_mean": round(float(cos.mean()), 4),
            "cos_vs_fp32_min": round(float(cos.min()), 4),
            "text_cos_vs_fp32": round(float(np.einsum("ij,ij->i", tembs[prec], tref).mean()), 4),
            "top10_overlap": round(float(overlap), 3),
        })
    return out


# ── 2. indexing ───────────────────────────────────────────────────────


def _load_baseline_worker():
    """Import the pre-Milestone-5 sequential worker from git (for comparison)."""
    try:
        src = subprocess.run(["git", "-C", str(ROOT), "show", f"{BASELINE_REV}:backend/app/indexer/worker.py"],
                             capture_output=True, text=True, check=True).stdout
    except Exception:
        return None
    path = Path(tempfile.mkdtemp()) / "baseline_worker.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("app.indexer._baseline_worker", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def run_index(files: list[Path], precision: str, ocr: str, worker: str = "pipelined",
              batch: int = 32, extra_env: dict | None = None) -> dict:
    from app import config
    from app.db.schema import get_db, init_db
    from app.embeddings import get_encoder, reset_encoder
    from app.indexer import ocr_engine

    tmp = Path(tempfile.mkdtemp(prefix="vs_bench_"))
    lib = tmp / "library"
    lib.mkdir()
    for i, p in enumerate(files):  # flat copy keeps the scan cost realistic
        shutil.copyfile(p, lib / f"{i:05d}_{p.name}")
    env = {"SEARCH_DATA_DIR": str(tmp / "data"), "VS_EMBED_BACKEND": "onnx", "VS_ONNX_PRECISION": precision,
           "VS_OCR_ENGINE": ocr, "VS_BATCH_SIZE": str(batch), **(extra_env or {})}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        config.reset_settings()
        reset_encoder()
        ocr_engine.reset_ocr_engine()
        try:
            from app.search import faiss_index
            faiss_index.reset_index()
        except Exception:
            pass
        init_db()
        get_encoder()  # model load is not part of indexing throughput
        with get_db() as conn:
            fid = conn.execute("INSERT INTO watched_folders (path) VALUES (?)", (str(lib),)).lastrowid
        if worker == "baseline":
            mod = _load_baseline_worker()
            if mod is None:
                return {}
            w = mod.IndexWorker()
        else:
            from app.indexer.worker import IndexWorker
            w = IndexWorker()
        t0 = time.perf_counter()
        asyncio.run(w.start([fid]))
        wall = time.perf_counter() - t0
        p = w.get_progress()
        with get_db() as conn:
            done = conn.execute("SELECT COUNT(*) FROM images WHERE status='COMPLETED'").fetchone()[0]
            words = conn.execute("SELECT COALESCE(SUM(word_count),0) FROM image_ocr").fetchone()[0]
        st = getattr(w, "stage_times", {})
        res = {
            "worker": worker, "precision": precision, "ocr": ocr_engine.get_ocr_engine().engine, "n": len(files),
            "completed": done, "failed": p.failed, "wall_s": round(wall, 1),
            "img_per_s": round(done / wall, 2), "ocr_words": int(words),
            "scan_s": round(st.get("scan_wall", 0.0), 1),
            "stage_cpu_s": {k: round(v, 1) for k, v in st.items() if k not in ("scan_wall", "hooks_wall")},
            "env": extra_env or {},
        }
        log(f"[index] {res}")
        return res
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        config.reset_settings()
        reset_encoder()
        ocr_engine.reset_ocr_engine()
        shutil.rmtree(tmp, ignore_errors=True)


# ── 3. FAISS ──────────────────────────────────────────────────────────


def bench_faiss(sizes: list[int], dim: int = 512, n_queries: int = 300, k: int = 60) -> dict:
    rng = np.random.default_rng(0)
    rows = []
    try:
        from app.search.faiss_index import VectorIndex
        impl = "app.search.faiss_index.VectorIndex"
    except Exception:
        VectorIndex = None
        impl = "faiss.IndexFlatIP (raw)"
    import faiss

    for n in sizes:
        # clustered vectors look more like real embeddings than iid noise
        centers = rng.standard_normal((64, dim)).astype(np.float32)
        mat = centers[rng.integers(0, 64, n)] + 0.6 * rng.standard_normal((n, dim)).astype(np.float32)
        mat /= np.linalg.norm(mat, axis=1, keepdims=True)
        ids = np.arange(1, n + 1, dtype=np.int64)
        t0 = time.perf_counter()
        if VectorIndex is not None:
            idx = VectorIndex(ids, mat, model="bench")
            kind = idx.index_type
            search = lambda q: idx.search(q, k)  # noqa: E731
        else:
            raw = faiss.IndexFlatIP(dim)
            raw.add(mat)
            kind = "IndexFlatIP"
            search = lambda q: raw.search(q, k)  # noqa: E731
        build_ms = (time.perf_counter() - t0) * 1000
        qs = mat[rng.integers(0, n, n_queries)] + 0.05 * rng.standard_normal((n_queries, dim)).astype(np.float32)
        qs /= np.linalg.norm(qs, axis=1, keepdims=True)
        for q in qs[:10]:
            search(q[None])
        lat = []
        for q in qs:
            t0 = time.perf_counter()
            search(q[None])
            lat.append((time.perf_counter() - t0) * 1000)
        # recall@10 of the index vs exact search
        exact = np.argsort(-(qs[:100] @ mat.T), axis=1)[:, :10] + 1
        got = [search(q[None])[0][0][:10] for q in qs[:100]]
        recall = float(np.mean([len(set(g.tolist()) & set(e.tolist())) / 10 for g, e in zip(got, exact)]))
        rows.append({"n": n, "index": kind, "build_ms": round(build_ms, 1), "p50_ms": round(pct(lat, 50), 3),
                     "p99_ms": round(pct(lat, 99), 3), "recall@10": round(recall, 3)})
        log(f"[faiss] {rows[-1]}")
    return {"impl": impl, "k": k, "rows": rows}


# ── 4. dedup ──────────────────────────────────────────────────────────


def bench_dedup(index_dir: Path, dataset: Path) -> dict | None:
    """Recall / precision of /duplicates on a full index of the chaos dataset."""
    gt_path = dataset / "ground_truth.json"
    if not (index_dir / "index.db").exists() or not gt_path.exists():
        return None
    from app import config
    from app.dedup import DedupParams, find_duplicates
    from app.dedup import verify as dverify
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_dedup import evaluate

    old = os.environ.get("SEARCH_DATA_DIR")
    os.environ["SEARCH_DATA_DIR"] = str(index_dir)
    config.reset_settings()
    try:
        gt = json.loads(gt_path.read_text())
        rows = []
        for label, kw in [("default (verified)", {}), ("pHash only", {"threshold": 1.01}),
                          ("cosine only", {"max_hamming": -1}),
                          ("raw rule, no verification", {"verify": False})]:
            dverify.clear_cache()
            t0 = time.perf_counter()
            payload = find_duplicates(DedupParams(**kw))
            cold = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            find_duplicates(DedupParams(**kw))
            warm = (time.perf_counter() - t0) * 1000
            ev = evaluate(payload, gt, dataset)
            rows.append({"label": label, "cold_ms": round(cold), "warm_ms": round(warm), **{
                k: ev[k] for k in ("recall", "recall_by_kind", "pair_precision", "n_groups", "n_images",
                                   "largest_groups", "recall_by_transform", "planted_pairs")}})
            log(f"[dedup] {rows[-1]}")
        return {"n_images": len(gt["files"]), "rows": rows}
    finally:
        if old is None:
            os.environ.pop("SEARCH_DATA_DIR", None)
        else:
            os.environ["SEARCH_DATA_DIR"] = old
        config.reset_settings()


# ── report ────────────────────────────────────────────────────────────


def render(info: dict, enc: dict | None, idx: list[dict], fa: dict | None, args, total_s: float,
           dd: dict | None = None) -> str:
    L = ["# Benchmarks", "",
         f"Generated by `python scripts/benchmark.py --dataset {args.dataset_label} --n {args.n}` "
         f"on {time.strftime('%Y-%m-%d')} (total benchmark runtime {total_s / 60:.1f} min).", "",
         "## Machine", "", "| | |", "|---|---|"]
    for k in ("cpu", "cpu_count", "memory", "load_avg_at_start", "platform", "python", "onnxruntime", "providers", "numpy", "faiss",
              "pillow", "tesseract"):
        L.append(f"| {k} | {info.get(k, '')} |")
    L.append("")
    L += ["The machine was shared with other agents/processes during the run (see load average), so "
          "absolute numbers carry some noise; relative comparisons were run back to back.", ""]
    L += ["No GPU; everything runs on the CPU. Model: `Xenova/clip-vit-base-patch32` (ViT-B/32, 512-d) "
          "vision + text towers exported to ONNX.", ""]
    if enc:
        r = enc["rows"]
        L += ["## 1. Encoder precision (fp32 / fp16 / int8)", "",
              f"{enc['n']} images from the chaos dataset, pre-decoded (≤1024 px), batch {enc['batch']}, "
              f"`intra_op_num_threads={os.cpu_count()}`. *Model* = ONNX vision tower only on pre-processed "
              "pixels; *end-to-end* = `encode_images()` incl. resize/crop/normalise. Agreement is measured "
              "against fp32 on the same images; top-10 overlap = mean |top10(prec) ∩ top10(fp32)| / 10 "
              f"over {len(QUERIES)} text queries.", "",
              "| precision | model img/s | end-to-end img/s | preprocess ms/img | text query p50 / p99 ms "
              "| image cos vs fp32 (mean / min) | text cos vs fp32 | top-10 overlap | load s |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for p in PRECISIONS:
            x = r[p]
            L.append(f"| **{p}** | {x['img_per_s_model']} | {x['img_per_s_e2e']} | {x['preprocess_ms_per_img']} | "
                     f"{x['text_ms_p50']} / {x['text_ms_p99']} | {x['cos_vs_fp32_mean']} / {x['cos_vs_fp32_min']} | "
                     f"{x['text_cos_vs_fp32']} | {x['top10_overlap']} | {x['load_s']} |")
        best = max(PRECISIONS, key=lambda p: r[p]["img_per_s_model"])
        L += ["", f"Fastest image encoder on this CPU: **{best}** "
              f"({r[best]['img_per_s_model'] / r['fp32']['img_per_s_model']:.2f}× fp32). "
              "fp16 has no fast kernels on x86 CPUs (ONNX Runtime inserts casts), so it is mainly useful "
              "for GPU / disk size. int8 (dynamic quantisation) is fast but lossy — compare its cosine / top-10 "
              "agreement with fp32 above before choosing it.", ""]
    if idx:
        L += ["## 2. Full indexing throughput", "",
              f"`IndexWorker` on a fresh `SEARCH_DATA_DIR` over {args.n_index} files copied from the dataset "
              "(scan + SHA-256 + thumbnail → decode → colour palette → OCR → pHash/dHash → batch embed → SQLite). "
              "Model load excluded. *Stage CPU-s* are summed over worker threads (they overlap in the pipelined "
              "worker, so they exceed wall time).", "",
              "| worker | precision | OCR | images | wall s | **img/s** | scan s | OCR words | stage CPU-s |",
              "|---|---|---|---:|---:|---:|---:|---:|---|"]
        for x in idx:
            if not x:
                continue
            st = ", ".join(f"{k} {v}" for k, v in x["stage_cpu_s"].items()) or "–"
            env = (" " + " ".join(f"{k}={v}" for k, v in x["env"].items())) if x["env"] else ""
            L.append(f"| {x['worker']}{env} | {x['precision']} | {x['ocr']} | {x['completed']}/{x['n']} | {x['wall_s']} | "
                     f"**{x['img_per_s']}** | {x['scan_s']} | {x['ocr_words']} | {st} |")
        L += ["", "*baseline* is the pre-M5 sequential worker (git `e30029a`), run in-process for comparison: "
              "one image at a time through colour → OCR, embedding every 32 images, one DB connection per step.", ""]
    if fa:
        L += ["## 3. FAISS query latency", "",
              f"`{fa['impl']}`, single-query top-{fa['k']} search, 512-d clustered unit vectors, 300 queries "
              "after warm-up; recall@10 vs exact brute force.", "",
              "| vectors | index | build ms | p50 ms | p99 ms | recall@10 |", "|---:|---|---:|---:|---:|---:|"]
        for x in fa["rows"]:
            L.append(f"| {x['n']:,} | {x['index']} | {x['build_ms']} | {x['p50_ms']} | {x['p99_ms']} | {x['recall@10']} |")
        L.append("")
    if dd:
        L += ["## 4. Duplicate detection", "",
              f"`GET /duplicates` logic (`app.dedup.find_duplicates`) on the full {dd['n_images']}-image chaos "
              f"index (`data/demo_index`, int8 CLIP), scored against the {dd['rows'][0]['planted_pairs']} planted "
              "pairs in `ground_truth.json` (5 % exact copies, 5 % near copies: resize / re-encode / crop / "
              "brightness, alone or combined). A pair is recalled when both files land in the same group; "
              "pair precision counts within-group pairs that belong to one planted family. *cold* = first call "
              "(pixel verification of all candidates), *warm* = repeat call (pair-score cache).", "",
              "| setting | recall | exact | near | pair precision | groups | images in groups | largest groups | cold ms | warm ms |",
              "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
        for r in dd["rows"]:
            k = r["recall_by_kind"]
            L.append(f"| {r['label']} | **{r['recall']:.3f}** | {k.get('exact', 0):.3f} | {k.get('near', 0):.3f} | "
                     f"{r['pair_precision']:.3f} | {r['n_groups']} | {r['n_images']} | {r['largest_groups']} | "
                     f"{r['cold_ms']} | {r['warm_ms']} |")
        L += ["", "Near-duplicate recall by transform (default setting): "
              + ", ".join(f"{k} {v}" for k, v in dd["rows"][0]["recall_by_transform"].items()), "",
              "On this synthetic library the raw PRD rule (pHash ≤ 6 OR cosine ≥ 0.95) chains thousands of "
              "same-template images (slides of one deck, screenshots of one app, generated forests) into a "
              "few giant groups: int8 CLIP puts them at cosine 0.95-0.98 while a 5 % crop of the same image "
              "can drop to ~0.90. The default therefore widens the cosine search to ≥ 0.88 and verifies "
              "every near candidate on the thumbnails (mean-chromaticity check + brightness-invariant colour "
              "correlation after a crop-alignment search, see `app/dedup/verify.py`).", ""]
    if getattr(args, "extra_md", None):
        L += [args.extra_md, ""]
    L += ["## Tuning knobs", "",
          "| env var | default | effect |", "|---|---|---|",
          "| `VS_ONNX_PRECISION` | `fp32` | `fp32` / `fp16` / `int8` model files |",
          "| `VS_BATCH_SIZE` | `32` | images per embed batch = pipeline chunk size |",
          "| `VS_ONNX_THREADS` | cpu_count | ONNX intra-op threads |",
          "| `VS_ONNX_INTER_THREADS`, `VS_ONNX_EXEC_MODE` | unset, `sequential` | ONNX inter-op parallelism |",
          "| `VS_ONNX_SPIN` | ORT default | `0` disables intra-op thread spinning (frees cores for OCR) |",
          "| `VS_INDEX_WORKERS` | cpu_count | decode/colour/OCR/hash thread-pool size |",
          "| `VS_OCR_ENGINE` | `auto` | `none` skips OCR (largest single cost) |", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=ROOT / "data" / "chaos")
    ap.add_argument("--n", type=int, default=400, help="images for the encoder benchmark")
    ap.add_argument("--n-index", type=int, default=None, help="images for indexing runs (default --n)")
    ap.add_argument("--baseline-n", type=int, default=200, help="images for the sequential baseline (0 = skip)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", default="", help="comma list of: encoders,index,faiss,dedup")
    ap.add_argument("--faiss-sizes", default="5000,20000,60000")
    ap.add_argument("--dedup-index", type=Path, default=ROOT / "data" / "demo_index",
                    help="data dir holding a full index of --dataset (from scripts/demo.py)")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "BENCHMARKS.md")
    ap.add_argument("--json", type=Path, default=None, help="also dump raw results as JSON")
    args = ap.parse_args(argv)
    args.n_index = args.n_index or args.n
    try:
        args.dataset_label = str(args.dataset.resolve().relative_to(ROOT))
    except ValueError:
        args.dataset_label = str(args.dataset)
    skip = set(filter(None, args.skip.split(",")))

    t_all = time.perf_counter()
    files = list_images(args.dataset)
    if not files:
        sys.exit(f"no images under {args.dataset}")
    rnd = random.Random(args.seed)
    sample = rnd.sample(files, min(args.n, len(files)))
    sample_idx = rnd.sample(files, min(args.n_index, len(files)))
    info = machine_info()
    log(json.dumps(info))

    enc = None if "encoders" in skip else bench_encoders(sample, args.batch)
    idx: list[dict] = []
    if "index" not in skip:
        idx.append(run_index(sample_idx, "int8", "none"))
        idx.append(run_index(sample_idx, "int8", "tesseract"))
        idx.append(run_index(sample_idx, "fp32", "tesseract"))
        if args.baseline_n:
            base = sample_idx[: args.baseline_n]
            idx.append(run_index(base, "int8", "none", worker="baseline"))
            idx.append(run_index(base, "int8", "tesseract", worker="baseline"))
            idx.append(run_index(base, "int8", "tesseract"))
    fa = None if "faiss" in skip else bench_faiss([int(s) for s in args.faiss_sizes.split(",") if s])
    dd = None if "dedup" in skip else bench_dedup(args.dedup_index.resolve(), args.dataset.resolve())

    total = time.perf_counter() - t_all
    md = render(info, enc, idx, fa, args, total, dd)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    if args.json:
        args.json.write_text(json.dumps({"machine": info, "encoders": enc, "index": idx, "faiss": fa, "dedup": dd},
                                        indent=1, default=str))
    log(f"wrote {args.out} in {total / 60:.1f} min")
    print(md)


if __name__ == "__main__":
    main()

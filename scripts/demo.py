#!/usr/bin/env python3
"""Scripted end-to-end demo: index → every query mode → galaxy → duplicates.

    python scripts/demo.py --dataset data/chaos                 # in-process (FastAPI TestClient)
    python scripts/demo.py --dataset data/chaos --url http://127.0.0.1:8000   # against a live server

Prints a readable transcript and saves it to ``docs/demo/demo_transcript.md``.
In-process runs keep their index in ``--data-dir`` (default ``data/demo_index``)
so a second run re-uses it (``--fresh`` starts over). Endpoints that are not
implemented yet (404 / 405 / 501) are noted and skipped, never fatal.
When the dataset has a ``ground_truth.json`` (``make_chaos_dataset.py``), hits
are annotated with their true category and precision@10 is reported.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("VS_MODELS_DIR", str(ROOT / "data" / "models"))

from PIL import Image, ImageDraw  # noqa: E402

MISSING = (404, 405, 501)


# ── HTTP client abstraction ───────────────────────────────────────────


class Api:
    """Tiny wrapper over a live server (requests) or an in-process TestClient."""

    def __init__(self, url: str | None) -> None:
        self.url = url.rstrip("/") if url else None
        self._ctx = None
        if self.url:
            import requests
            self._s = requests.Session()
        else:
            from fastapi.testclient import TestClient
            from app.main import app
            self._ctx = TestClient(app)
            self._s = self._ctx.__enter__()

    def close(self) -> None:
        if self._ctx is not None:
            self._ctx.__exit__(None, None, None)

    def _full(self, path: str) -> str:
        return f"{self.url}{path}" if self.url else path

    def call(self, method: str, path: str, **kw):
        t0 = time.perf_counter()
        r = self._s.request(method, self._full(path), timeout=600, **kw) if self.url else \
            self._s.request(method, path, **kw)
        ms = (time.perf_counter() - t0) * 1000
        try:
            body = r.json()
        except Exception:
            body = r.text
        return r.status_code, body, ms


# ── transcript ────────────────────────────────────────────────────────


class Transcript:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, line: str = "") -> None:
        print(line, flush=True)
        self.lines.append(line)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n")


# ── helpers ───────────────────────────────────────────────────────────


def sketch_png() -> str:
    """A quick 'sunset over mountains' doodle, as a data URL."""
    img = Image.new("RGB", (512, 512), "white")
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, 512, 300), fill=(250, 150, 60))
    d.ellipse((200, 120, 300, 220), fill=(255, 230, 150))
    d.polygon([(0, 380), (130, 220), (260, 360), (380, 200), (512, 340), (512, 512), (0, 512)], fill=(60, 50, 90))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class Demo:
    def __init__(self, api: Api, dataset: Path, out: Transcript, k: int) -> None:
        self.api, self.dataset, self.t, self.k = api, dataset.resolve(), out, k
        gt_path = dataset / "ground_truth.json"
        self.gt = json.loads(gt_path.read_text()) if gt_path.exists() else None
        self.results: dict[str, dict] = {}

    def rel(self, p: str) -> str:
        try:
            return str(Path(p).resolve().relative_to(self.dataset))
        except ValueError:
            return p

    def category(self, p: str) -> str:
        if not self.gt:
            return ""
        return self.gt["files"].get(self.rel(p), {}).get("category", "?")

    def gt_text(self, p: str) -> str:
        if not self.gt:
            return ""
        return self.gt["files"].get(self.rel(p), {}).get("text", "")

    def all_images(self) -> dict[int, str]:
        """image_id → file_path via the paginated ``GET /images`` (limit ≤ 200)."""
        if getattr(self, "_paths", None):
            return self._paths
        out: dict[int, str] = {}
        offset = 0
        while True:
            st, body, _ = self.api.call("GET", "/api/v1/images", params={"limit": 200, "offset": offset})
            if st != 200 or not body.get("images"):
                break
            out.update({i["image_id"]: i["file_path"] for i in body["images"]})
            offset += len(body["images"])
            if offset >= body.get("total", 0):
                break
        self._paths = out
        return out

    # ── steps ─────────────────────────────────────────────────────────
    def step(self, title: str) -> None:
        self.t("")
        self.t(f"## {title}")
        self.t("")

    def missing(self, what: str, status: int, body) -> bool:
        if status in MISSING:
            self.t(f"> ⚠️ `{what}` is not available yet (HTTP {status}) — skipped.")
            return True
        if status >= 400:
            self.t(f"> ❌ `{what}` failed: HTTP {status} {str(body)[:300]}")
            return True
        return False

    def register_and_index(self, force: bool) -> None:
        self.step("1. Register folder & index")
        st, body, _ = self.api.call("POST", "/api/v1/folders", json={"path": str(self.dataset)})
        if st == 409:
            _, folders, _ = self.api.call("GET", "/api/v1/folders")
            fid = next(f["folder_id"] for f in folders if Path(f["path"]) == self.dataset)
            self.t(f"Folder already registered → folder_id={fid}")
        elif st >= 400:
            raise SystemExit(f"could not register folder: {st} {body}")
        else:
            fid = body["folder_id"]
            self.t(f"`POST /folders` → folder_id={fid} ({self.dataset})")
        self.folder_id = fid
        _, info, _ = self.api.call("GET", "/api/v1/system/info")
        if isinstance(info, dict):
            keep = {k: info[k] for k in ("encoder", "embed_backend", "ocr_engine", "onnx_precision", "device") if k in info}
            self.t(f"`GET /system/info` → {json.dumps(keep or info)[:400]}")
        t0 = time.time()
        st, body, _ = self.api.call("POST", "/api/v1/index/start", json={"folder_ids": [fid], "force_reindex": force})
        self.t(f"`POST /index/start` → {body}")
        last, stable = None, 0
        progress_lines = []
        while True:
            time.sleep(2.0)
            _, stats, _ = self.api.call("GET", "/api/v1/stats")
            bs = stats.get("by_status", {})
            total = stats.get("total_images", 0)
            pending = sum(v for k, v in bs.items() if k not in ("COMPLETED", "ERROR"))
            snap = (total, bs.get("COMPLETED", 0), bs.get("ERROR", 0))
            line = (f"  t={time.time() - t0:6.1f}s  images={total:5d}  completed={snap[1]:5d}  "
                    f"error={snap[2]:3d}  in-flight={pending:5d}")
            print(line, flush=True)
            progress_lines.append(line)
            stable = stable + 1 if snap == last and pending == 0 else 0
            last = snap
            if total and pending == 0 and stable >= 2:
                break
        wall = time.time() - t0
        shown = progress_lines[:: max(1, len(progress_lines) // 12)] + progress_lines[-1:]
        self.t("```")
        for line in shown:
            self.t(line)
        self.t("```")
        self.t(f"Indexed **{last[1]}** images ({last[2]} errors) in **{wall:.1f}s** "
               f"→ {last[1] / max(wall, 1e-9):.1f} img/s end-to-end (scan + colour + OCR + hashes + CLIP + DB).")
        self.results["index"] = {"completed": last[1], "errors": last[2], "wall_s": round(wall, 1)}

    def show_hits(self, name: str, body: dict, ms: float, expect: str | None = None,
                  expect_text: str | None = None) -> None:
        hits = body.get("results", [])
        self.t(f"took_ms={body.get('took_ms', ms):.1f} (client {ms:.0f} ms), faiss_ms={body.get('faiss_ms', 0):.3f}, "
               f"modes={body.get('modes')}, candidates={body.get('total_candidates')}")
        self.t("")
        self.t("| # | score | file | category | scores |")
        self.t("|---:|---:|---|---|---|")
        for i, h in enumerate(hits[:self.k], 1):
            sc = ", ".join(f"{k}={v:.3f}" for k, v in (h.get("scores") or {}).items() if v is not None)
            self.t(f"| {i} | {h['score']:.3f} | `{self.rel(h['file_path'])}` | {self.category(h['file_path'])} | {sc} |")
        res = {"took_ms": body.get("took_ms"), "n": len(hits)}
        if self.gt and hits:
            top = hits[:10]
            if expect:
                p = sum(self.category(h["file_path"]) in expect.split("|") for h in top) / len(top)
                res["p@10"] = p
                self.t(f"\nprecision@10 (category ∈ {{{expect}}}): **{p:.2f}**")
            if expect_text:
                p = sum(expect_text.lower() in self.gt_text(h["file_path"]).lower() for h in top) / len(top)
                res["p@10"] = p
                self.t(f"\nprecision@10 (ground-truth text contains “{expect_text}”): **{p:.2f}**")
        self.results[name] = res

    def search(self, name: str, title: str, payload: dict, expect: str | None = None,
               expect_text: str | None = None) -> dict | None:
        self.t(f"### {title}")
        self.t(f"`POST /search` {json.dumps({k: (v if k != 'sketch_png' else v[:40] + '…') for k, v in payload.items()})}")
        st, body, ms = self.api.call("POST", "/api/v1/search", json={"k": max(self.k, 10), **payload})
        if self.missing("POST /search", st, body):
            return None
        self.show_hits(name, body, ms, expect, expect_text)
        self.t("")
        return body

    def queries(self) -> None:
        self.step("2. Query modes")
        vibes = [
            ("text_sunset", "sunset over the mountains", "photo_sky"),
            ("text_city", "a city skyline at night with lit windows", "photo_city"),
            ("text_forest", "a quiet pine forest", "photo_forest"),
            ("text_meme", "a funny meme with a caption", "meme"),
            ("text_code", "source code in a dark text editor", "code"),
            ("text_chart", "a bar chart of quarterly revenue", "chart"),
            ("text_sketch", "a child's pencil doodle on white paper", "sketch"),
            ("text_ui", "a mobile app settings screen", "ui"),
        ]
        for name, q, exp in vibes:
            self.search(name, f"Text / vibe: “{q}”", {"text": q}, expect=exp)
        self.search("ocr_gradient", "Embedded text (OCR): “gradient descent”", {"ocr": "gradient descent"},
                    expect_text="gradient descent")
        self.search("ocr_compiles", "Embedded text (OCR): “code compiles”", {"ocr": "code compiles"},
                    expect_text="code compiles")
        self.search("color_forest", "Colour palette: forest greens", {"colors": ["#145A1E", "#3C6E28", "#96C8E6"]},
                    expect="photo_forest|meme")
        self.search("color_night", "Colour palette: night blues + window yellow", {"colors": ["#0A0C1E", "#FFDC78"]},
                    expect="photo_city|meme")
        self.search("sketch", "Sketch: sunset-over-mountains doodle", {"sketch_png": sketch_png()},
                    expect="photo_sky|meme|palette")
        self.search("hybrid", "Hybrid: text “lecture slide” + OCR “transformers” + white palette",
                    {"text": "a lecture slide", "ocr": "transformers", "colors": ["#FFFFFF"],
                     "weights": {"semantic": 0.4, "ocr": 0.4, "color": 0.2}}, expect_text="transformers")

        # image → image on a planted near-duplicate
        self.t("### Image-to-image: find similar to an indexed image")
        if not self.gt:
            self.t("(no ground truth — skipped)")
            return
        pair = next((p for p in self.gt["duplicate_pairs"] if p["kind"] == "near"), None)
        by_rel = {self.rel(p): i for i, p in self.all_images().items()}
        qid = by_rel.get(pair["a"]) if pair else None
        if qid is None:
            self.t("(query image not found in /images listing — skipped)")
            return
        st, body, ms = self.api.call("GET", f"/api/v1/search/similar/{qid}", params={"k": max(self.k, 10)})
        self.t(f"`GET /search/similar/{qid}` (query `{pair['a']}`, planted near-dup `{pair['b']}` via {pair.get('transform')})")
        if not self.missing("GET /search/similar", st, body):
            self.show_hits("similar", body, ms)
            rank = next((i for i, h in enumerate(body.get("results", []), 1) if self.rel(h["file_path"]) == pair["b"]), None)
            self.t(f"\nPlanted near-duplicate rank: **{rank}**")
            self.results["similar"]["dup_rank"] = rank
        self.t("")
        self.t("### Image-to-image: uploaded file")
        with open(self.dataset / pair["b"], "rb") as f:
            st, body, ms = self.api.call("POST", "/api/v1/search/upload", files={"file": (Path(pair["b"]).name, f.read())},
                                         data={"k": str(max(self.k, 10))})
        self.t(f"`POST /search/upload` file=`{pair['b']}`")
        if not self.missing("POST /search/upload", st, body):
            self.show_hits("upload", body, ms)
            rank = next((i for i, h in enumerate(body.get("results", []), 1) if self.rel(h["file_path"]) == pair["a"]), None)
            self.t(f"\nOriginal image rank: **{rank}**")

    def galaxy(self) -> None:
        self.step("3. Galaxy (UMAP 2D + clusters)")
        t0 = time.time()
        st, body, ms = self.api.call("POST", "/api/v1/galaxy/compute", json={"method": "umap", "dims": 2})
        if self.missing("POST /galaxy/compute", st, body):
            return
        clusters = body.get("clusters", [])
        self.t(f"`POST /galaxy/compute` → {len(body.get('points', []))} points, {len(clusters)} clusters in {time.time() - t0:.1f}s")
        self.t("")
        cats_by_cluster: dict[int, dict[str, int]] = {}
        if self.gt:
            path_of = self.all_images()
            for pt in body.get("points", []):
                c = cats_by_cluster.setdefault(pt.get("cluster_id"), {})
                cat = self.category(path_of.get(pt["image_id"], ""))
                c[cat] = c.get(cat, 0) + 1
        self.t("| cluster | label | size | dominant true categories |")
        self.t("|---:|---|---:|---|")
        purity_num = 0
        for c in sorted(clusters, key=lambda c: -c.get("size", 0)):
            cats = cats_by_cluster.get(c["cluster_id"], {})
            top = sorted(cats.items(), key=lambda kv: -kv[1])[:3]
            purity_num += top[0][1] if top else 0
            self.t(f"| {c['cluster_id']} | {c.get('label')} | {c.get('size')} | "
                   + ", ".join(f"{k} {v}" for k, v in top) + " |")
        n = len(body.get("points", []))
        if self.gt and n:
            self.t(f"\nCluster purity vs. true categories: **{purity_num / n:.2f}**")
            self.results["galaxy"] = {"clusters": len(clusters), "purity": round(purity_num / n, 3)}

    def duplicates(self) -> None:
        self.step("4. Duplicates")
        st, body, ms = self.api.call("GET", "/api/v1/duplicates", params={"threshold": 0.95, "max_hamming": 6})
        if self.missing("GET /duplicates", st, body):
            return
        kinds: dict[str, int] = {}
        for g in body["groups"]:
            kinds[g["kind"]] = kinds.get(g["kind"], 0) + 1
        self.t(f"`GET /duplicates?threshold=0.95&max_hamming=6` → {body['n_groups']} groups "
               f"({', '.join(f'{v} {k}' for k, v in kinds.items())}), {body['n_images']} images, {ms:.0f} ms")
        self.t("")
        for g in body["groups"][:6]:
            self.t(f"- group {g['group_id']} ({g['kind']}): " + "; ".join(
                f"`{self.rel(i['file_path'])}` sim={i['similarity']}" for i in g["images"][:4]))
        if self.gt:
            from eval_dedup import evaluate
            ev = evaluate(body, self.gt, self.dataset)
            self.t("")
            self.t(f"Recall on planted duplicates: **{ev['recall']:.3f}** (by kind: {ev['recall_by_kind']}); "
                   f"pair precision **{ev['pair_precision']:.3f}** over {ev['within_group_pairs']} within-group pairs.")
            self.t(f"Near-dup recall by transform: {ev['recall_by_transform']}")
            self.results["dedup"] = {k: ev[k] for k in ("recall", "recall_by_kind", "pair_precision", "n_groups")}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=ROOT / "data" / "chaos")
    ap.add_argument("--url", default=None, help="live server base URL (default: in-process TestClient)")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "demo_index")
    ap.add_argument("--fresh", action="store_true", help="wipe --data-dir first (in-process only)")
    ap.add_argument("--force-reindex", action="store_true")
    ap.add_argument("--precision", default=os.environ.get("VS_ONNX_PRECISION", "int8"))
    ap.add_argument("--k", type=int, default=8, help="hits shown per query")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "demo" / "demo_transcript.md")
    a = ap.parse_args(argv)

    if not a.url:
        if a.fresh and a.data_dir.exists():
            shutil.rmtree(a.data_dir)
        os.environ["SEARCH_DATA_DIR"] = str(a.data_dir.resolve())
        os.environ.setdefault("VS_EMBED_BACKEND", "onnx")
        os.environ["VS_ONNX_PRECISION"] = a.precision
    t = Transcript()
    t("# PixelIndex — scripted end-to-end demo")
    t("")
    t(f"Generated by `python scripts/demo.py --dataset {a.dataset}`"
      f"{' --url ' + a.url if a.url else ''} on {time.strftime('%Y-%m-%d %H:%M')}. "
      f"Mode: {'live server ' + a.url if a.url else 'in-process FastAPI TestClient'}; "
      f"encoder backend `{os.environ.get('VS_EMBED_BACKEND', 'onnx')}` / `{a.precision}`.")
    api = Api(a.url)
    demo = Demo(api, a.dataset, t, a.k)
    t0 = time.time()
    try:
        demo.register_and_index(a.force_reindex)
        demo.queries()
        demo.galaxy()
        demo.duplicates()
    finally:
        api.close()
    demo.step("Summary")
    t("```json")
    t(json.dumps(demo.results, indent=1, default=str))
    t("```")
    t(f"\nTotal demo time {time.time() - t0:.0f}s.")
    t.save(a.out)
    print(f"\ntranscript → {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

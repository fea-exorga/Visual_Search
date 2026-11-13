# PixelIndex — Technical PRD & Execution Plan

> Zero-Text Multimodal Desktop Search · Reinforce SST idea `idea_84e84c6c`
> This document turns the product PRD (`/PRD`) into an engineering plan:
> milestone labels, acceptance criteria, interface contracts and the agent
> swarm that executes it.

---

## 1. Goal

A **local-first** search engine for messy image folders (lecture screenshots,
memes, photos) that finds images by:

| Query mode | Input | Engine |
|---|---|---|
| Vibe / semantic | free text ("cozy rainy evening") | CLIP text tower → FAISS |
| Image-to-image | an indexed image or an uploaded file | CLIP vision tower → FAISS |
| Sketch-to-image | live sketch-pad PNG | CLIP vision tower + stroke colour/structure heuristics |
| Colour palette | 1–5 hex swatches | Lab-space palette distance on stored KMeans palettes |
| Embedded text | keywords | SQLite FTS5 over Tesseract/EasyOCR text |
| **Hybrid** | any combination of the above | weighted fusion |

…plus a **2D/3D galaxy explorer** (UMAP/t-SNE + clustering) and
**duplicate detection**, with **no cloud calls** at query or index time.

## 2. Non-goals

* No cloud sync, accounts, or multi-user support.
* No video / PDF indexing (raster images only: PNG, JPEG, WebP, BMP).
* No model training or fine-tuning.

## 3. Key technical decisions

| Decision | Choice | Why |
|---|---|---|
| Default encoder | **CLIP ViT-B/32 on ONNX Runtime** (`Xenova/clip-vit-base-patch32`, 512-d) | Torch-free install (~60 MB of wheels vs ~3 GB), fast CPU inference, ships fp32 / fp16 / int8 exports → covers the M5 quantisation goal directly. |
| Alternative encoders | `torch` backend (any HF CLIP/SigLIP, e.g. `google/siglip-base-patch16-224`), `fake` backend (tests) | One `Encoder` protocol, chosen by `VS_EMBED_BACKEND`. |
| Vector storage | `image_embeddings` table in SQLite (float32 blob per image) | Single source of truth; no row-offset drift. FAISS is a rebuildable cache. |
| ANN index | FAISS `IndexFlatIP` (exact cosine) ≤ 50k vectors, `IndexHNSWFlat` above | Exact search is already < 1 ms for a few thousand 512-d vectors. |
| OCR | Tesseract by default (`auto` → tesseract → easyocr → none) | No torch dependency; EasyOCR remains optional. |
| Projection | UMAP (default) and t-SNE; PCA pre-reduction to 50-d | UMAP keeps global structure and is fast; t-SNE kept for comparison. |
| Clustering | KMeans with k picked by silhouette score in [4, 24] on the 2D/3D coordinates (HDBSCAN optional) | Deterministic, every point is assigned to a cluster. |
| Cluster names | Zero-shot: CLIP text embeddings of a vibe vocabulary vs cluster centroid + top OCR keywords | No LLM, fully local. |
| Frontend | React 18 + TS + Vite + Tailwind (existing) + `three` for 3D galaxy, Canvas2D for 2D galaxy and sketch-pad | Smooth with thousands of points. |

## 4. Milestones (labels)

Every milestone has a GitHub label `milestone-N` and a tracking issue.

### `milestone-1` — Local Directory Indexer ✅ (foundation hardened)
* Recursive scan, SHA-256 delta detection, thumbnails.
* KMeans dominant colour palette (5 swatches).
* OCR via Tesseract / EasyOCR → `image_ocr` + FTS5 (triggers keep it in sync).
* CLIP/SigLIP image embeddings via the pluggable `Encoder` (ONNX default).
* **Acceptance:** `pytest tests/test_indexer.py` green; every image of the
  synthetic dataset reaches `COMPLETED` with a palette and a unit-norm 512-d vector;
  exact-duplicate files reuse the donor's features.

### `milestone-2` — Local Vector Index & Cross-Modal Retrieval
* `app/search/faiss_index.py`: build/load/save FAISS index from `image_embeddings`,
  id-mapped (`IndexIDMap2`), auto-rebuilt by a post-index hook, thread-safe singleton.
* Query modes: text, image-id, uploaded image, sketch (base64 PNG), colours, OCR keywords.
* `app/search/hybrid.py`: score fusion `S = w_sem·sem + w_ocr·ocr + w_col·col`
  (each component min-max/bounded to [0,1]; weights renormalised over the
  components the query actually uses).
* **Acceptance:** FAISS top-k latency < 1 ms p50 on 5k vectors (measured & reported in
  the response as `faiss_ms`); text "red" ranks red images first with the fake encoder;
  image-to-image returns the query image's near-duplicate in the top 3.

### `milestone-3` — Manifold Projection & Theme Clustering
* `app/projection/`: UMAP & t-SNE to 2D and 3D, normalised to [-1, 1].
* KMeans clustering + zero-shot cluster labels, stable cluster colours.
* Results persisted to `image_projections` / `clusters`; recompute endpoint.
* **Acceptance:** every embedded image gets coordinates for each computed
  (method, dims); clusters of the synthetic dataset separate solid colours from text slides.

### `milestone-4` — Interactive Frontend & Galaxy Explorer
* Search bar with mode chips, hybrid filters (OCR keywords, colour swatches, cluster tags, weights).
* Live sketch-pad (brush, colours, eraser, clear, debounced live query).
* Galaxy explorer: 2D (Canvas2D, pan/zoom, hover thumbnail) and 3D (three.js, orbit),
  coloured by cluster, click → inspector, search results highlighted.
* "Find similar" from the inspector, duplicates view.
* **Acceptance:** `npm run build` passes with zero TS errors; Playwright smoke
  screenshot of each view against a running backend.

### `milestone-5` — Optimisation, Dedup & Demo
* ONNX precision switch (fp32 / fp16 / int8), batched + pipelined indexing
  (thread-pool decode/colour/OCR overlapping with batch encode), benchmark script.
* Duplicate detection: exact (SHA-256), near (pHash Hamming ≤ 6 **or** cosine ≥ 0.95 via FAISS range search), grouped with union-find.
* `scripts/make_chaos_dataset.py`: ≥ 3,000 synthetic "chaotic" images
  (slides with text, memes with captions, photos-like gradients, sketches, screenshots,
  resized/re-encoded duplicates).
* `scripts/benchmark.py` → `docs/BENCHMARKS.md`; `scripts/demo.py` scripted end-to-end
  demo (index → each query mode → galaxy → dedup) with recorded output.
* **Acceptance:** benchmark table for the 3 precisions; dedup recall ≥ 0.9 on the
  planted duplicates; demo log + screenshots committed under `docs/demo/`.

## 5. Architecture

```
Frontend (React/Vite)  ──REST/SSE──►  FastAPI (app/main.py + app/api/*)
                                         │
     ┌────────────────────────┬──────────┼───────────────┬───────────────────┐
     ▼                        ▼          ▼               ▼                   ▼
 indexer/ (M1)          search/ (M2)  projection/ (M3)  dedup/ (M5)     embeddings/ (M1/M5)
 scan→colour→OCR→embed  FAISS+hybrid  UMAP/t-SNE+KMeans  pHash+cosine   ONNX CLIP / torch / fake
     │                        │          │               │
     └───────────── SQLite (WAL, FTS5): images, image_colors, image_ocr,
                     image_embeddings, image_projections, clusters, image_hashes
```

## 6. Shared contracts (do not change without updating this doc)

### 6.1 Encoder (`app/embeddings`)
```python
enc = get_encoder()           # singleton; backend from VS_EMBED_BACKEND
enc.name: str                 # e.g. "clip-vit-base-patch32/onnx-int8"
enc.dim: int                  # 512
enc.encode_images(list[PIL.Image]) -> np.ndarray  # (N, dim) float32, L2-normalised
enc.encode_texts(list[str])        -> np.ndarray  # (N, dim) float32, L2-normalised
```

### 6.2 Vectors (`app/db/vectors.py`)
`save_vectors(ids, mat, model)`, `load_vectors(model=None, image_ids=None) -> (ids int64, mat)`,
`get_vector(image_id)`, `current_models()`. Only `COMPLETED` images are loaded.

### 6.3 Post-index hooks (`app/indexer/hooks.py`)
`register_post_index_hook(name, fn)` — run after every indexing job (FAISS rebuild, dedup hashes).

### 6.4 HTTP API (prefix `/api/v1`)

Existing (M1): `POST/GET/DELETE /folders`, `POST /index/start|pause|resume`,
`GET /index/stream` (SSE), `GET /images`, `GET /images/{id}`, `GET /thumbnails/{id}`,
`GET /stats`, `GET /system/info`.

New:

```
POST /search                    (M2)
  body: { text?: str, image_id?: int, sketch_png?: str (data URL or base64),
          colors?: [ "#RRGGBB" ], ocr?: str, cluster_ids?: [int], folder_id?: int,
          projection?: {method, dims}, k?: int = 60,
          weights?: { semantic?: float, ocr?: float, color?: float } }
  → { results: [SearchHit], took_ms: float, faiss_ms: float, modes: [str], total_candidates: int }

SearchHit = { image_id, score, scores: {semantic?, ocr?, color?}, file_path,
              thumbnail_url, width, height, colors: [ColorSwatch], ocr_snippet, cluster_id? }

POST /search/upload  (multipart field "file", optional form "k")   (M2) → same as /search
GET  /search/similar/{image_id}?k=24                                (M2) → same as /search
POST /search/rebuild                                                (M2) → { n_vectors, index_type, build_ms }

GET  /galaxy?method=umap|tsne&dims=2|3                               (M3)
  → { method, dims, points: [{image_id, x, y, z?, cluster_id}],
      clusters: [{cluster_id, label, size, color, centroid: [x, y, z?]}], computed_at }
  404 if not computed yet.
POST /galaxy/compute  body {method="umap", dims=2, n_neighbors?, min_dist?, perplexity?, n_clusters?}
  → same payload as GET (synchronous; fast enough for ~10k points)

GET  /duplicates?threshold=0.95&max_hamming=6                         (M5)
  → { groups: [{ group_id, kind: "exact"|"near",
                 images: [{image_id, file_path, thumbnail_url, file_size, width, height, similarity}] }],
      n_groups, n_images }
```

## 7. Agent swarm (execution)

Foundation (encoders, vector store, schema, hooks, router stubs, test
harness) is built first by the orchestrator, then four agents run **in parallel
on disjoint files**:

| Agent | Label | Owns (write access) | Tests |
|---|---|---|---|
| `search-agent` | `milestone-2` | `app/search/**`, `app/api/search.py` | `tests/test_search.py` |
| `galaxy-agent` | `milestone-3` | `app/projection/**`, `app/api/galaxy.py` | `tests/test_projection.py` |
| `frontend-agent` | `milestone-4` | `frontend/**` | `npm run build` |
| `perf-agent` | `milestone-5` | `app/dedup/**`, `app/api/dedup.py`, `app/indexer/worker.py`, `app/embeddings/onnx_clip.py`, `scripts/**`, `docs/BENCHMARKS.md` | `tests/test_dedup.py`, `tests/test_worker_perf.py` |

Orchestrator then integrates: full test suite, real-model end-to-end run on the
chaos dataset, Playwright screenshots, README, commit & push.

## 8. Success metrics

| Metric | Target |
|---|---|
| FAISS query latency (5k × 512-d) | < 1 ms p50 |
| End-to-end `/search` latency (text query, CPU) | < 100 ms p50 (excluding first model load) |
| Indexing throughput (CPU, int8, OCR on) | reported in `docs/BENCHMARKS.md` |
| Dedup recall on planted duplicates | ≥ 0.9 |
| Galaxy render | ≥ 3k points interactive |

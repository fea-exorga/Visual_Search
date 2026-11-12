# 🔍 PixelIndex — Zero-Text Multimodal Desktop Search

**Milestone 1: Local Directory Indexer**

A local image indexing pipeline that scans directories, extracts OCR text, computes dominant color palettes, and generates CLIP/SigLIP vision embeddings — all with zero cloud dependencies.

> **Organization**: [Reinforce AI/ML Club at Scaler School of Technology](https://github.com/Reinforce-SST/Reinforce_Club-SST)  
> **Project**: Zero-Text Multimodal Desktop Search ([idea_84e84c6c](https://www.reinforce-sst.com/dashboard/ideas/idea_84e84c6c))

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     3-Pane Desktop UI                          │
│  ┌──────────┐  ┌─────────────────────┐  ┌──────────────────┐  │
│  │ Sidebar  │  │   Center Canvas     │  │ Inspector Drawer │  │
│  │ (260px)  │  │   Asset Grid + SSE  │  │ (340px)          │  │
│  └──────────┘  └─────────────────────┘  └──────────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ REST + SSE
┌──────────────────────────▼──────────────────────────────────────┐
│                    FastAPI Backend                               │
│  POST /folders │ POST /index/start │ GET /index/stream (SSE)   │
│  GET /images   │ GET /images/{id}  │ GET /thumbnails/{id}      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────┐
    │           Indexing Pipeline                  │
    │  1. File Scan → 2. Color Palette (KMeans)   │
    │  3. OCR Text (EasyOCR) → 4. Vision Embed    │
    │        (SigLIP / CLIP)                       │
    └──────────────────────┬──────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────┐
    │  SQLite (WAL + FTS5)  │  embeddings.npy     │
    └─────────────────────────────────────────────┘
```

## Quick Start

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 — the Vite dev server proxies `/api` to `localhost:8000`.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| File Scanner | `os.scandir()` recursive + SHA-256 delta detection |
| Color Extraction | scikit-learn `MiniBatchKMeans` (k=5, 64×64 downsample) |
| OCR Engine | EasyOCR (English, GPU-accelerated) |
| Vision Embeddings | Google SigLIP (`siglip-base-patch16-224`, 768-d) |
| Database | SQLite3 (WAL mode) + FTS5 full-text search |
| API Server | FastAPI + Server-Sent Events (SSE) |
| Frontend | React 18 + TypeScript + Vite + Tailwind CSS |

## Project Structure

```
backend/
├── app/
│   ├── config.py              # Settings dataclass & singleton
│   ├── main.py                # FastAPI app + REST + SSE endpoints
│   ├── db/
│   │   └── schema.py          # SQLite schema, WAL, FTS5 triggers
│   └── indexer/
│       ├── scanner.py         # os.scandir, SHA-256, delta detection
│       ├── color_extractor.py # KMeans dominant color palette
│       ├── ocr_engine.py      # EasyOCR wrapper + FTS5 sync
│       ├── embedder.py        # SigLIP batched vision embeddings
│       └── worker.py          # Pipeline orchestrator + progress
└── requirements.txt

frontend/
├── src/
│   ├── App.tsx                # 3-pane layout root
│   ├── main.tsx               # React entry point
│   ├── components/
│   │   ├── Sidebar.tsx        # Folder list + pipeline settings
│   │   ├── CenterPanel.tsx    # Progress banner + image grid
│   │   ├── ImageGrid.tsx      # Paginated responsive grid
│   │   ├── ImageCard.tsx      # Thumbnail + color bar + OCR badge
│   │   └── InspectorDrawer.tsx # 4-section feature inspector
│   ├── hooks/
│   │   ├── useApi.ts          # REST API wrapper functions
│   │   └── useSSE.ts          # SSE streaming hook
│   └── types/
│       └── index.ts           # TypeScript interfaces
├── index.html
├── vite.config.ts
├── tailwind.config.js
└── package.json
```

## License

Part of the Reinforce SST research project. See organization for details.

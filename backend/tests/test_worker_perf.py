"""Milestone 5 — pipelined indexing worker."""

import asyncio

import numpy as np
import pytest

from app.db.schema import get_db
from app.db.vectors import load_vectors
from tests.conftest import image_id_for


def _register(path):
    with get_db() as conn:
        return conn.execute("INSERT INTO watched_folders (path) VALUES (?)", (str(path.resolve()),)).lastrowid


@pytest.fixture()
def small_batches(monkeypatch):
    from app import config
    monkeypatch.setenv("VS_BATCH_SIZE", "3")  # several chunks → exercises the pipeline overlap
    config.reset_settings()


def test_pipelined_worker_indexes_everything(data_dir, dataset, small_batches):
    from app.indexer.worker import IndexWorker

    fid = _register(dataset["solid_red"].parent)
    snapshots = []
    w = IndexWorker(progress_callback=lambda p: snapshots.append((p.color_done, p.embedded, p.current_stage)))
    asyncio.run(w.start([fid]))
    p = w.get_progress()

    with get_db() as conn:
        statuses = [r["status"] for r in conn.execute("SELECT status FROM images")]
        n_hash = conn.execute("SELECT COUNT(*) c FROM image_hashes WHERE phash IS NOT NULL").fetchone()["c"]
        n_ocr = conn.execute("SELECT COUNT(*) c FROM image_ocr").fetchone()["c"]
        n_col = conn.execute("SELECT COUNT(DISTINCT image_id) c FROM image_colors").fetchone()["c"]
    assert set(statuses) == {"COMPLETED"} and len(statuses) == len(dataset)
    # within one scan the exact dup's donor is not COMPLETED yet, so both are processed
    assert p.total_files == len(dataset)
    assert p.color_done == p.ocr_done == p.embedded == p.total_files
    assert p.failed == 0 and p.skipped == 0
    assert p.current_stage == "done" and not p.is_running
    assert n_hash == len(dataset) and n_col == len(dataset) and n_ocr == len(dataset)
    # progress was reported incrementally and monotonically
    embedded = [s[1] for s in snapshots]
    assert embedded == sorted(embedded) and len(set(embedded)) > 2
    ids, vecs = load_vectors()
    assert len(ids) == len(dataset)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)
    for k in ("decode", "color", "hash", "embed"):
        assert w.stage_times.get(k, 0) > 0


def test_pipelined_matches_direct_encoding(data_dir, dataset, small_batches):
    """Vectors from the pipeline equal encoding the decoded file directly."""
    from PIL import Image
    from app.embeddings import get_encoder
    from app.db.vectors import get_vector
    from app.indexer.worker import IndexWorker

    fid = _register(dataset["solid_red"].parent)
    asyncio.run(IndexWorker().start([fid]))
    enc = get_encoder()
    for key in ("slide", "circle_green", "solid_yellow"):
        v = get_vector(image_id_for(dataset[key]))
        ref = enc.encode_images([Image.open(dataset[key]).convert("RGB")])[0]
        assert float(v @ ref) > 0.999


def test_cancel_between_chunks(data_dir, dataset, small_batches):
    from app.indexer.worker import IndexWorker

    fid = _register(dataset["solid_red"].parent)
    holder = {}

    def cb(p):
        if p.embedded >= 1:
            holder["w"].cancel()

    w = IndexWorker(progress_callback=cb)
    holder["w"] = w
    asyncio.run(w.start([fid]))
    p = w.get_progress()
    assert p.current_stage == "cancelled" and not p.is_running
    with get_db() as conn:
        counts = dict(conn.execute("SELECT status, COUNT(*) FROM images GROUP BY status").fetchall())
    assert counts.get("PENDING", 0) > 0          # work left undone
    assert counts.get("COMPLETED", 0) >= 1       # finished chunks were persisted
    assert p.embedded < p.total_files
    # resuming with a fresh worker finishes the job
    w2 = IndexWorker()
    with get_db() as conn:
        pending = [r[0] for r in conn.execute("SELECT image_id FROM images WHERE status='PENDING'")]
    asyncio.run(w2._run_pipeline(pending, 3))
    with get_db() as conn:
        assert {r[0] for r in conn.execute("SELECT status FROM images")} == {"COMPLETED"}


def test_pause_resume(data_dir, dataset, small_batches):
    from app.indexer.worker import IndexWorker

    fid = _register(dataset["solid_red"].parent)
    state = {"paused_seen": False}

    async def run():
        w = IndexWorker()

        def cb(p):
            if p.embedded >= 1 and not state["paused_seen"]:
                state["paused_seen"] = True
                w.pause()
                asyncio.get_running_loop().call_later(0.2, w.resume)

        w._callback = cb
        await w.start([fid])
        return w

    w = asyncio.run(run())
    assert state["paused_seen"]
    p = w.get_progress()
    assert p.current_stage == "done" and p.embedded == p.total_files and not p.is_paused


def test_corrupt_image_marks_error(data_dir, dataset, small_batches, monkeypatch):
    """A file that breaks after scanning → ERROR with a message; the rest completes."""
    from app.indexer import worker as worker_mod
    from app.indexer.worker import IndexWorker

    real_scan = worker_mod.scan_directory
    victim = dataset["circle_green"]

    def scan_then_corrupt(path, fid):
        ids = real_scan(path, fid)
        victim.write_bytes(b"\x89PNG\r\n\x1a\n this is not really a png")
        return ids

    monkeypatch.setattr(worker_mod, "scan_directory", scan_then_corrupt)
    fid = _register(dataset["solid_red"].parent)
    w = IndexWorker()
    asyncio.run(w.start([fid]))
    p = w.get_progress()
    bad = image_id_for(victim)
    with get_db() as conn:
        row = conn.execute("SELECT status, error_message FROM images WHERE image_id=?", (bad,)).fetchone()
        others = {r[0] for r in conn.execute("SELECT status FROM images WHERE image_id != ?", (bad,))}
    assert row["status"] == "ERROR" and row["error_message"]
    assert others == {"COMPLETED"}
    assert p.failed == 1 and p.errors[0]["image_id"] == bad
    assert p.embedded == p.total_files - 1
    assert p.current_stage == "done"


def test_corrupt_file_skipped_by_scanner(data_dir, dataset, small_batches):
    """A file that is corrupt from the start never blocks the job."""
    from app.indexer.worker import IndexWorker

    (dataset["solid_red"].parent / "broken.png").write_bytes(b"not an image at all")
    fid = _register(dataset["solid_red"].parent)
    w = IndexWorker()
    asyncio.run(w.start([fid]))
    with get_db() as conn:
        statuses = {r[0] for r in conn.execute("SELECT status FROM images")}
    assert statuses == {"COMPLETED"} and w.get_progress().current_stage == "done"

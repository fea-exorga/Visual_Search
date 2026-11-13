#!/usr/bin/env python3
"""Generate a chaotic, realistic-looking personal image library.

    python scripts/make_chaos_dataset.py --out data/chaos --n 3000 --seed 0

Categories: lecture slides (real CS/ML text), code screenshots, memes with
captions, photo-like scenes (sunsets, mountains, sea, night city, forest),
abstract colour palettes, hand-drawn sketches, bar/line charts and UI
screenshots. Random sizes / aspect ratios, PNG/JPEG/WebP, nested messy
folders and file names (``IMG_4021.png``, ``Screenshot_2026-11-05 (3).png``).

~5 % exact duplicates (byte copies) and ~5 % near duplicates (resize,
re-encode, crop, brightness) are planted. ``ground_truth.json`` records the
category (and text, when any) of every file plus all duplicate pairs.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

# ── fonts ─────────────────────────────────────────────────────────────

_FONT_CANDIDATES = {
    "sans": ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/freefont/FreeSans.ttf"],
    "bold": ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"],
    "mono": ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/usr/share/fonts/truetype/freefont/FreeMono.ttf"],
    "serif": ["/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"],
}
_font_cache: dict = {}


def font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    key = (kind, size)
    if key not in _font_cache:
        f = None
        for path in _FONT_CANDIDATES.get(kind, []):
            if os.path.exists(path):
                f = ImageFont.truetype(path, size)
                break
        _font_cache[key] = f or ImageFont.load_default(size=size)
    return _font_cache[key]


# ── vocabulary ────────────────────────────────────────────────────────

TOPICS = [
    "Gradient Descent", "Backpropagation", "Convolutional Neural Networks", "Dijkstra's Algorithm",
    "Hash Tables", "Big-O Notation", "Attention Is All You Need", "Recurrent Neural Networks",
    "Support Vector Machines", "Dynamic Programming", "Binary Search Trees", "TCP/IP Networking",
    "Process Scheduling", "Bayesian Inference", "Principal Component Analysis", "K-Means Clustering",
    "Reinforcement Learning", "Overfitting and Regularization", "Graph Traversal BFS DFS",
    "Database Normalization", "Linear Regression", "Logistic Regression", "Decision Trees",
    "Random Forests", "Transformers", "Diffusion Models", "Word Embeddings", "Batch Normalization",
    "Virtual Memory", "Compilers and Parsing", "Public Key Cryptography", "Sorting Algorithms",
    "Merge Sort", "Quick Sort", "Heaps and Priority Queues", "Markov Chains", "Monte Carlo Methods",
    "Cross Validation", "Loss Functions", "Contrastive Learning", "Vision Transformers",
    "Distributed Systems", "Consensus Raft Paxos", "Cache Coherence", "Information Theory",
]
BULLETS = [
    "learning rate controls step size", "minimise the loss function", "chain rule of derivatives",
    "time complexity O(n log n)", "space complexity O(n)", "weights are shared across positions",
    "softmax over attention scores", "use a validation set", "early stopping prevents overfitting",
    "convexity guarantees a global minimum", "amortised constant time", "collisions handled by chaining",
    "the kernel trick maps to higher dimensions", "eigenvectors of the covariance matrix",
    "expected reward under the policy", "stochastic mini batches", "residual connections help",
    "precision vs recall trade-off", "normalise the input features", "exam next week, chapter 4",
]
MEME_TOP = [
    "WHEN THE CODE COMPILES", "ME AT 3AM", "NOBODY:", "ONE DOES NOT SIMPLY", "MY GPU WHEN",
    "WHEN THE PROF SAYS", "DEBUGGING PRODUCTION", "WHEN YOU FIX ONE BUG", "POV:", "MONDAY MORNING",
    "WHEN THE DEADLINE IS TOMORROW", "ME EXPLAINING MY CODE", "STACK OVERFLOW BE LIKE",
]
MEME_BOTTOM = [
    "ON THE FIRST TRY", "AND IT WORKS", "MERGE TO MAIN", "TEN NEW ONES APPEAR", "IT WAS A TYPO",
    "WHAT IS SLEEP", "THIS WILL BE ON THE EXAM", "IT WORKS ON MY MACHINE", "TRAINING FOR 3 DAYS",
    "GRADIENT EXPLODED", "COFFEE IS LIFE", "SEMICOLON MISSING", "JUST ONE MORE EPOCH",
]
CODE_LINES = [
    "def train(model, loader, epochs=10):", "    for epoch in range(epochs):",
    "        for x, y in loader:", "            loss = criterion(model(x), y)",
    "            loss.backward()", "            optimizer.step()", "import numpy as np",
    "from sklearn.cluster import KMeans", "class Node:", "    def __init__(self, key):",
    "        self.left = self.right = None", "return sorted(items, key=lambda x: x[1])",
    "const app = express();", "app.get('/api/v1/search', handler);", "SELECT * FROM images WHERE id = 42;",
    "if (err) { console.error(err); }", "for (int i = 0; i < n; i++) {", "    sum += a[i] * b[i];",
    "}", "# TODO: fix this hack", "print(f'accuracy: {acc:.3f}')", "x = torch.randn(32, 3, 224, 224)",
    "while queue:", "    node = queue.popleft()", "fn main() { println!(\"hello\"); }",
]
UI_LABELS = ["Settings", "Profile", "Messages", "Search", "Dashboard", "Upload", "Cancel", "Save",
             "Log out", "Notifications", "Inbox", "Files", "Calendar", "Help", "Sign in"]
CHART_TITLES = ["Quarterly Revenue", "Training Loss", "Validation Accuracy", "Monthly Users",
                "Latency p99 (ms)", "Sales by Region", "GPU Utilisation", "Exam Scores"]

SIZES = [(1920, 1080), (1280, 720), (1366, 768), (1024, 768), (800, 600), (1080, 1920),
         (750, 1334), (1200, 1200), (640, 480), (1600, 900), (2048, 1536), (512, 512), (900, 1200)]

FOLDERS = [
    "Downloads", "Downloads/old downloads", "Desktop", "Desktop/stuff", "Desktop/stuff/misc (2)",
    "Pictures/Camera Roll", "Pictures/2025/July", "Pictures/2026/trip", "Screenshots",
    "Screenshots/2026-11", "uni/ML course/week 3", "uni/ML course/week 7/slides", "uni/algorithms",
    "memes", "memes/good ones", "memes/good ones/reaction", "WhatsApp Images", "random",
    "backup (copy)/Desktop", "Documents/scans", "projects/vision/assets", "projects/web/mockups",
    "wallpapers", "art/sketches", "charts export",
]

CATEGORY_WEIGHTS = {
    "slide": 16, "code": 10, "meme": 12, "photo_sky": 10, "photo_city": 7, "photo_forest": 7,
    "palette": 8, "sketch": 9, "chart": 10, "ui": 11,
}


# ── helpers ───────────────────────────────────────────────────────────


def rand_color(rng: random.Random, lo: int = 0, hi: int = 255) -> tuple[int, int, int]:
    return tuple(rng.randint(lo, hi) for _ in range(3))  # type: ignore[return-value]


def pick_size(rng: random.Random, kind: str) -> tuple[int, int]:
    if kind in ("slide", "ui", "code", "chart"):
        w, h = rng.choice([(1920, 1080), (1280, 720), (1366, 768), (1600, 900), (1024, 768), (2560, 1440)])
    else:
        w, h = rng.choice(SIZES)
    s = rng.uniform(0.45, 1.0)
    return max(240, int(w * s)), max(240, int(h * s))


def vgradient(w: int, h: int, top, bottom) -> Image.Image:
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    arr = (1 - t) * np.array(top, np.float32) + t * np.array(bottom, np.float32)
    return Image.fromarray(np.broadcast_to(arr, (h, w, 3)).astype(np.uint8))


def add_noise(img: Image.Image, rng: random.Random, amount: float) -> Image.Image:
    if amount <= 0:
        return img
    nrng = np.random.default_rng(rng.randint(0, 2**31))
    arr = np.asarray(img, dtype=np.int16)
    arr = arr + nrng.normal(0, amount, arr.shape).astype(np.int16)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def fit_text(draw: ImageDraw.ImageDraw, text: str, kind: str, max_w: int, size: int) -> ImageFont.FreeTypeFont:
    while size > 12:
        f = font(kind, size)
        if draw.textlength(text, font=f) <= max_w:
            return f
        size = int(size * 0.88)
    return font(kind, 12)


# ── generators ────────────────────────────────────────────────────────


def gen_slide(rng: random.Random) -> tuple[Image.Image, str]:
    w, h = pick_size(rng, "slide")
    theme = rng.choice(["light", "light", "dark", "blue", "uni"])
    bg = {"light": (250, 250, 248), "dark": (30, 32, 40), "blue": (18, 52, 110), "uni": (255, 255, 255)}[theme]
    fg = (20, 20, 20) if theme in ("light", "uni") else (240, 240, 240)
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    if theme == "uni":
        d.rectangle((0, 0, w, int(h * 0.14)), fill=rand_color(rng, 0, 140))
        fg_title = (255, 255, 255)
    else:
        fg_title = fg
    topic = rng.choice(TOPICS)
    lecture = rng.randint(1, 14)
    title = f"Lecture {lecture}: {topic}" if rng.random() < 0.5 else topic
    tf = fit_text(d, title, "bold", int(w * 0.88), int(h * 0.075))
    d.text((int(w * 0.05), int(h * 0.035)), title, fill=fg_title, font=tf)
    bullets = rng.sample(BULLETS, rng.randint(2, 5))
    bf = font("sans", max(14, int(h * 0.042)))
    y = int(h * 0.24)
    for b in bullets:
        d.text((int(w * 0.07), y), "• " + b, fill=fg, font=bf)
        y += int(h * 0.09)
    if rng.random() < 0.4:  # a small diagram
        x0, y0 = int(w * 0.68), int(h * 0.55)
        for k in range(rng.randint(3, 6)):
            cx, cy = x0 + rng.randint(0, int(w * 0.25)), y0 + rng.randint(0, int(h * 0.35))
            r = int(h * 0.03)
            d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=fg, width=3)
    d.text((w - int(w * 0.08), h - int(h * 0.07)), str(rng.randint(2, 60)), fill=fg, font=font("sans", max(12, int(h * 0.03))))
    return img, " ".join([title] + bullets)


def gen_code(rng: random.Random) -> tuple[Image.Image, str]:
    w, h = pick_size(rng, "code")
    bg = rng.choice([(30, 30, 30), (40, 44, 52), (39, 40, 34), (0, 43, 54), (250, 250, 250)])
    light = bg[0] > 200
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    fs = max(13, int(h * rng.uniform(0.022, 0.035)))
    f = font("mono", fs)
    palette = [(198, 120, 221), (97, 175, 239), (152, 195, 121), (229, 192, 123), (224, 108, 117), (171, 178, 191)]
    if light:
        palette = [(0, 0, 160), (160, 0, 120), (0, 120, 0), (140, 80, 0), (60, 60, 60)]
    d.rectangle((0, 0, w, int(fs * 1.8)), fill=tuple(max(0, c - 12) for c in bg) if not light else (225, 225, 225))
    d.text((10, 4), rng.choice(["train.py", "main.rs", "server.js", "query.sql", "utils.py", "index.ts"]),
           fill=(150, 150, 150), font=font("sans", int(fs * 0.9)))
    y, n, lines = int(fs * 2.4), 1, []
    start = rng.randint(0, len(CODE_LINES) - 1)
    while y < h - fs:
        line = CODE_LINES[(start + n) % len(CODE_LINES)]
        lines.append(line)
        d.text((8, y), f"{n:>3}", fill=(110, 110, 110), font=f)
        x = 8 + d.textlength("0000 ", font=f)
        for tok in line.split(" "):
            col = rng.choice(palette)
            d.text((x, y), tok, fill=col, font=f)
            x += d.textlength(tok + " ", font=f)
        y += int(fs * 1.45)
        n += 1
    return img, " ".join(lines[:6])


def gen_scene(rng: random.Random, kind: str, w: int | None = None, h: int | None = None) -> Image.Image:
    if w is None or h is None:
        w, h = pick_size(rng, kind)
    if kind == "photo_sky":
        mood = rng.choice(["sunset", "day", "dusk", "dawn"])
        top, bottom = {
            "sunset": ((70, 40, 120), (250, 150, 60)), "day": ((40, 110, 210), (170, 210, 250)),
            "dusk": ((20, 20, 60), (200, 90, 120)), "dawn": ((120, 160, 220), (250, 210, 170)),
        }[mood]
        top = tuple(int(np.clip(c + rng.randint(-25, 25), 0, 255)) for c in top)
        bottom = tuple(int(np.clip(c + rng.randint(-25, 25), 0, 255)) for c in bottom)
        img = vgradient(w, h, top, bottom)
        d = ImageDraw.Draw(img)
        horizon = int(h * rng.uniform(0.55, 0.75))
        if rng.random() < 0.8:  # sun
            sx, sy, r = rng.randint(int(w * .15), int(w * .85)), rng.randint(int(h * .15), horizon), int(min(w, h) * rng.uniform(.04, .1))
            glow = Image.new("RGB", (w, h), (0, 0, 0))
            ImageDraw.Draw(glow).ellipse((sx - 3 * r, sy - 3 * r, sx + 3 * r, sy + 3 * r), fill=(120, 80, 30))
            img = Image.fromarray(np.clip(np.asarray(img, np.int16) + np.asarray(glow.filter(ImageFilter.GaussianBlur(r * 1.5)), np.int16), 0, 255).astype(np.uint8))
            d = ImageDraw.Draw(img)
            d.ellipse((sx - r, sy - r, sx + r, sy + r), fill=(255, 235, 170))
        if rng.random() < 0.6:  # mountains
            for layer in range(rng.randint(1, 3)):
                base = horizon + layer * int(h * 0.04)
                pts = [(0, h)]
                x = 0
                while x < w:
                    pts.append((x, base - rng.randint(int(h * .05), int(h * .3))))
                    x += rng.randint(int(w * .08), int(w * .2))
                pts += [(w, base - rng.randint(0, int(h * .1))), (w, h)]
                shade = 40 + layer * 25
                d.polygon(pts, fill=(shade, shade + rng.randint(0, 20), shade + 30))
        else:  # sea
            sea_top = (20, 60, 110) if mood == "day" else (40, 30, 70)
            d.rectangle((0, horizon, w, h), fill=sea_top)
            for k in range(60):
                y = rng.randint(horizon, h)
                x = rng.randint(0, w)
                d.line((x, y, x + rng.randint(10, 80), y), fill=tuple(min(255, c + 60) for c in sea_top), width=2)
        return add_noise(img.filter(ImageFilter.GaussianBlur(0.8)), rng, 5)
    if kind == "photo_city":
        img = vgradient(w, h, (8, 10, 30), (40, 30, 70))
        d = ImageDraw.Draw(img)
        for _ in range(rng.randint(40, 120)):  # stars
            x, y = rng.randint(0, w), rng.randint(0, h // 2)
            d.point((x, y), fill=(220, 220, 255))
        x = 0
        while x < w:
            bw = rng.randint(int(w * .04), int(w * .12))
            bh = rng.randint(int(h * .2), int(h * .75))
            col = rng.randint(15, 40)
            d.rectangle((x, h - bh, x + bw, h), fill=(col, col, col + 10))
            for wy in range(h - bh + 8, h - 8, 14):
                for wx in range(x + 5, x + bw - 6, 11):
                    if rng.random() < 0.45:
                        d.rectangle((wx, wy, wx + 5, wy + 7), fill=rng.choice([(255, 220, 120), (255, 240, 180), (200, 220, 255)]))
            x += bw + rng.randint(0, 6)
        return add_noise(img.filter(ImageFilter.GaussianBlur(0.6)), rng, 6)
    # forest
    img = vgradient(w, h, (150, 200, 230), (200, 220, 200))
    d = ImageDraw.Draw(img)
    d.rectangle((0, int(h * .7), w, h), fill=(50, 90, 40))
    for _ in range(rng.randint(15, 45)):
        tx = rng.randint(-20, w + 20)
        th = rng.randint(int(h * .25), int(h * .6))
        tw = th // 3
        base = int(h * rng.uniform(.72, .98))
        g = rng.randint(60, 120)
        d.rectangle((tx - tw // 10, base - th // 5, tx + tw // 10, base), fill=(80, 50, 30))
        d.polygon([(tx, base - th), (tx - tw, base - th // 5), (tx + tw, base - th // 5)], fill=(20, g, 30))
    return add_noise(img.filter(ImageFilter.GaussianBlur(1.0)), rng, 7)


def gen_meme(rng: random.Random) -> tuple[Image.Image, str]:
    w, h = rng.choice([(600, 600), (800, 800), (720, 900), (1000, 750), (500, 500)])
    kind = rng.choice(["photo_sky", "photo_city", "photo_forest", "plain", "face"])
    if kind == "plain":
        img = Image.new("RGB", (w, h), rand_color(rng, 40, 220))
    elif kind == "face":
        img = Image.new("RGB", (w, h), rand_color(rng, 120, 230))
        d = ImageDraw.Draw(img)
        cx, cy, r = w // 2, h // 2, min(w, h) // 3
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 205, 60), outline=(0, 0, 0), width=4)
        d.ellipse((cx - r // 2, cy - r // 3, cx - r // 4, cy), fill=(0, 0, 0))
        d.ellipse((cx + r // 4, cy - r // 3, cx + r // 2, cy), fill=(0, 0, 0))
        d.arc((cx - r // 2, cy, cx + r // 2, cy + r // 2), 0, 180, fill=(0, 0, 0), width=5)
    else:
        img = gen_scene(rng, kind, w, h)
    d = ImageDraw.Draw(img)
    top, bottom = rng.choice(MEME_TOP), rng.choice(MEME_BOTTOM)
    for text, y in ((top, int(h * 0.04)), (bottom, int(h * 0.84))):
        f = fit_text(d, text, "bold", int(w * 0.92), int(h * 0.1))
        tw = d.textlength(text, font=f)
        d.text(((w - tw) / 2, y), text, font=f, fill="white", stroke_width=max(2, f.size // 14), stroke_fill="black")
    return img, f"{top} {bottom}"


def gen_palette(rng: random.Random) -> Image.Image:
    w, h = pick_size(rng, "palette")
    style = rng.choice(["stripes", "blocks", "blobs", "gradient"])
    cols = [rand_color(rng) for _ in range(rng.randint(3, 6))]
    img = Image.new("RGB", (w, h), cols[0])
    d = ImageDraw.Draw(img)
    if style == "stripes":
        for i, c in enumerate(cols):
            d.rectangle((i * w // len(cols), 0, (i + 1) * w // len(cols), h), fill=c)
    elif style == "blocks":
        for _ in range(rng.randint(6, 20)):
            x, y = rng.randint(0, w), rng.randint(0, h)
            d.rectangle((x, y, x + rng.randint(40, w // 2), y + rng.randint(40, h // 2)), fill=rng.choice(cols))
    elif style == "blobs":
        for _ in range(rng.randint(5, 15)):
            x, y, r = rng.randint(0, w), rng.randint(0, h), rng.randint(20, min(w, h) // 3)
            d.ellipse((x - r, y - r, x + r, y + r), fill=rng.choice(cols))
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0, 12)))
    else:
        if rng.random() < 0.5:
            img = vgradient(w, h, cols[0], cols[1])
        else:  # horizontal gradient
            img = vgradient(h, w, cols[0], cols[1]).transpose(Image.Transpose.ROTATE_90)
    return img


def gen_sketch(rng: random.Random) -> Image.Image:
    w, h = rng.choice([(800, 600), (1024, 768), (600, 600), (768, 1024)])
    paper = rng.choice([(255, 255, 255), (250, 246, 235), (245, 245, 245)])
    img = Image.new("RGB", (w, h), paper)
    d = ImageDraw.Draw(img)
    ink = rng.choice([(20, 20, 20), (30, 40, 140), (60, 60, 60), (150, 20, 20)])
    lw = rng.randint(2, 6)

    def jline(pts):
        jit = [(x + rng.uniform(-3, 3), y + rng.uniform(-3, 3)) for x, y in pts]
        d.line(jit, fill=ink, width=lw, joint="curve")

    for _ in range(rng.randint(1, 3)):
        what = rng.choice(["house", "sun", "stick", "smiley", "tree", "scribble", "cat"])
        cx, cy, s = rng.randint(w // 5, 4 * w // 5), rng.randint(h // 5, 4 * h // 5), rng.randint(min(w, h) // 8, min(w, h) // 4)
        if what == "house":
            jline([(cx - s, cy), (cx - s, cy + s), (cx + s, cy + s), (cx + s, cy), (cx - s, cy)])
            jline([(cx - s, cy), (cx, cy - s), (cx + s, cy)])
            jline([(cx - s // 4, cy + s), (cx - s // 4, cy + s // 3), (cx + s // 4, cy + s // 3), (cx + s // 4, cy + s)])
        elif what in ("sun", "smiley"):
            pts = [(cx + s * math.cos(t / 20 * 2 * math.pi), cy + s * math.sin(t / 20 * 2 * math.pi)) for t in range(21)]
            jline(pts)
            if what == "sun":
                for k in range(8):
                    a = k / 8 * 2 * math.pi
                    jline([(cx + 1.2 * s * math.cos(a), cy + 1.2 * s * math.sin(a)), (cx + 1.6 * s * math.cos(a), cy + 1.6 * s * math.sin(a))])
            else:
                d.ellipse((cx - s // 3, cy - s // 3, cx - s // 5, cy - s // 6), fill=ink)
                d.ellipse((cx + s // 5, cy - s // 3, cx + s // 3, cy - s // 6), fill=ink)
                d.arc((cx - s // 2, cy - s // 6, cx + s // 2, cy + s // 2), 20, 160, fill=ink, width=lw)
        elif what == "stick":
            d.ellipse((cx - s // 4, cy - s, cx + s // 4, cy - s // 2), outline=ink, width=lw)
            jline([(cx, cy - s // 2), (cx, cy + s // 3)])
            jline([(cx - s // 2, cy - s // 5), (cx + s // 2, cy - s // 5)])
            jline([(cx - s // 3, cy + s), (cx, cy + s // 3), (cx + s // 3, cy + s)])
        elif what == "tree":
            jline([(cx, cy + s), (cx, cy)])
            pts = [(cx + s * .6 * math.cos(t / 16 * 2 * math.pi), cy - s * .4 + s * .5 * math.sin(t / 16 * 2 * math.pi)) for t in range(17)]
            jline(pts)
        elif what == "cat":
            pts = [(cx + s * .6 * math.cos(t / 16 * 2 * math.pi), cy + s * .5 * math.sin(t / 16 * 2 * math.pi)) for t in range(17)]
            jline(pts)
            jline([(cx - s * .5, cy - s * .2), (cx - s * .4, cy - s * .8), (cx - s * .15, cy - s * .45)])
            jline([(cx + s * .5, cy - s * .2), (cx + s * .4, cy - s * .8), (cx + s * .15, cy - s * .45)])
        else:
            x, y = cx, cy
            pts = []
            for _ in range(rng.randint(10, 40)):
                x += rng.randint(-s // 3, s // 3)
                y += rng.randint(-s // 3, s // 3)
                pts.append((x, y))
            jline(pts)
    return img


def gen_chart(rng: random.Random) -> tuple[Image.Image, str]:
    w, h = pick_size(rng, "chart")
    img = Image.new("RGB", (w, h), rng.choice([(255, 255, 255), (248, 249, 250), (25, 25, 30)]))
    dark = img.getpixel((0, 0))[0] < 100
    fg = (230, 230, 230) if dark else (40, 40, 40)
    d = ImageDraw.Draw(img)
    title = rng.choice(CHART_TITLES)
    d.text((int(w * .06), int(h * .03)), title, fill=fg, font=font("bold", max(16, int(h * .06))))
    x0, y0, x1, y1 = int(w * .1), int(h * .18), int(w * .95), int(h * .88)
    d.line((x0, y0, x0, y1, x1, y1), fill=fg, width=3)
    n = rng.randint(4, 12)
    vals = [rng.uniform(0.1, 1.0) for _ in range(n)]
    col = rand_color(rng, 40, 230)
    if rng.random() < 0.5:
        bw = (x1 - x0) / n
        for i, v in enumerate(vals):
            d.rectangle((x0 + i * bw + bw * .15, y1 - v * (y1 - y0), x0 + (i + 1) * bw - bw * .15, y1), fill=col)
        kind = "bar chart"
    else:
        for s in range(rng.randint(1, 3)):
            vals = [min(1, max(0.02, vals[0] + rng.uniform(-.2, .2) * k / n)) for k in range(n)] if s else vals
            pts = [(x0 + i * (x1 - x0) / (n - 1), y1 - v * (y1 - y0)) for i, v in enumerate(vals)]
            d.line(pts, fill=rand_color(rng, 40, 230) if s else col, width=4)
        kind = "line chart"
    lf = font("sans", max(11, int(h * .03)))
    for i in range(5):
        y = y1 - i * (y1 - y0) / 4
        d.text((x0 - int(w * .07), y - 8), f"{i * 25}", fill=fg, font=lf)
    return img, f"{title} {kind}"


def gen_ui(rng: random.Random) -> tuple[Image.Image, str]:
    w, h = pick_size(rng, "ui")
    dark = rng.random() < 0.35
    bg = (32, 33, 36) if dark else (243, 244, 246)
    card = (48, 49, 54) if dark else (255, 255, 255)
    fg = (230, 230, 230) if dark else (30, 30, 30)
    accent = rand_color(rng, 30, 220)
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    bar = int(h * .06)
    d.rectangle((0, 0, w, bar), fill=accent)
    app = rng.choice(["PixelNotes", "TaskBoard", "CloudDrive", "MailBox", "StudyPlanner", "PhotoVault"])
    d.text((16, bar * .2), app, fill=(255, 255, 255), font=font("bold", int(bar * .55)))
    side = int(w * .18)
    d.rectangle((0, bar, side, h), fill=card)
    labels = rng.sample(UI_LABELS, 6)
    lf = font("sans", max(12, int(h * .028)))
    for i, lab in enumerate(labels):
        d.text((18, bar + 24 + i * int(h * .065)), lab, fill=fg, font=lf)
    for r in range(rng.randint(2, 4)):
        for c in range(rng.randint(2, 3)):
            cx0 = side + 20 + c * int((w - side) / 3)
            cy0 = bar + 20 + r * int(h * .22)
            d.rounded_rectangle((cx0, cy0, cx0 + int((w - side) / 3) - 20, cy0 + int(h * .19)), 12, fill=card)
            d.text((cx0 + 12, cy0 + 12), rng.choice(UI_LABELS), fill=fg, font=lf)
            d.rounded_rectangle((cx0 + 12, cy0 + int(h * .12), cx0 + 110, cy0 + int(h * .16)), 8, fill=accent)
    return img, " ".join([app] + labels)


def generate(kind: str, rng: random.Random) -> tuple[Image.Image, str]:
    if kind == "slide":
        return gen_slide(rng)
    if kind == "code":
        return gen_code(rng)
    if kind == "meme":
        return gen_meme(rng)
    if kind in ("photo_sky", "photo_city", "photo_forest"):
        return gen_scene(rng, kind), ""
    if kind == "palette":
        return gen_palette(rng), ""
    if kind == "sketch":
        return gen_sketch(rng), ""
    if kind == "chart":
        return gen_chart(rng)
    return gen_ui(rng)


# ── naming / saving ───────────────────────────────────────────────────


def pick_format(rng: random.Random, kind: str) -> str:
    if kind in ("slide", "code", "ui", "chart"):
        return rng.choices(["png", "jpg", "webp"], [7, 2, 1])[0]
    if kind.startswith("photo"):
        return rng.choices(["jpg", "png", "webp"], [7, 1, 2])[0]
    return rng.choices(["jpg", "png", "webp"], [4, 4, 2])[0]


def pick_name(rng: random.Random, kind: str, ext: str) -> str:
    y, m, dd = rng.choice([2024, 2025, 2026]), rng.randint(1, 12), rng.randint(1, 28)
    hh, mm, ss = rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59)
    choices = [
        f"IMG_{rng.randint(1000, 9999)}",
        f"Screenshot_{y}-{m:02d}-{dd:02d} ({rng.randint(1, 9)})",
        f"Screenshot {y}-{m:02d}-{dd:02d} at {hh}.{mm:02d}.{ss:02d}",
        f"Screen Shot {y}-{m:02d}-{dd:02d} at {hh}.{mm:02d}.{ss:02d} PM",
        f"download ({rng.randint(1, 40)})", f"image{rng.randint(0, 99)}", "untitled",
        f"Untitled design ({rng.randint(1, 30)})", f"WhatsApp Image {y}-{m:02d}-{dd:02d} at {hh}.{mm:02d}.{ss:02d}",
        f"PXL_{y}{m:02d}{dd:02d}_{hh:02d}{mm:02d}{ss:02d}{rng.randint(100, 999)}",
        f"photo_{y}-{m:02d}-{dd:02d}_{hh:02d}-{mm:02d}-{ss:02d}",
        f"{rng.randint(10**17, 10**18)}",
    ]
    by_kind = {
        "slide": [f"lecture{rng.randint(1, 14)}_slide{rng.randint(1, 60)}", f"week{rng.randint(1, 12)} slides p{rng.randint(1, 40)}"],
        "meme": ["meme_final_FINAL", f"lol ({rng.randint(1, 20)})", "this is so me", f"reaction_{rng.randint(1, 99)}"],
        "chart": [f"chart_export_{rng.randint(1, 99)}", "figure_1", f"plot({rng.randint(1, 9)})"],
        "sketch": [f"doodle_{rng.randint(1, 99)}", "scan0001", "drawing"],
    }
    pool = choices + by_kind.get(kind, [])
    return f"{rng.choice(pool)}.{ext}"


def encode(img: Image.Image, ext: str, rng: random.Random) -> bytes:
    buf = io.BytesIO()
    if ext == "jpg":
        img.save(buf, "JPEG", quality=rng.randint(75, 95))
    elif ext == "webp":
        img.save(buf, "WEBP", quality=rng.randint(70, 92), method=2)
    else:
        img.save(buf, "PNG", compress_level=rng.choice([1, 3, 6]))
    return buf.getvalue()


def near_dup(img: Image.Image, rng: random.Random) -> tuple[Image.Image, str]:
    ops = rng.sample(["resize", "reencode", "crop", "brightness"], rng.randint(1, 2))
    out = img
    for op in ops:
        if op == "resize":
            s = rng.uniform(0.45, 0.85)
            out = out.resize((max(64, int(out.width * s)), max(64, int(out.height * s))), Image.BILINEAR)
        elif op == "crop":
            fx, fy = rng.uniform(0.02, 0.08), rng.uniform(0.02, 0.08)
            out = out.crop((int(out.width * fx), int(out.height * fy), int(out.width * (1 - fx)), int(out.height * (1 - fy))))
        elif op == "brightness":
            out = ImageEnhance.Brightness(out).enhance(rng.choice([rng.uniform(0.8, 0.92), rng.uniform(1.08, 1.2)]))
        # reencode: handled by saving as low-quality JPEG below
    return out, "+".join(sorted(ops))


# ── worker processes ──────────────────────────────────────────────────


def _make_base(args) -> dict:
    i, seed, out, kind, rel = args
    rng = random.Random(seed * 1_000_003 + i)
    img, text = generate(kind, rng)
    data = encode(img, rel.rsplit(".", 1)[1], rng)
    (out / rel).parent.mkdir(parents=True, exist_ok=True)
    (out / rel).write_bytes(data)
    return {"path": rel, "category": kind, "text": text, "width": img.width, "height": img.height}


def _make_near(args) -> dict:
    j, seed, out, src, rel = args
    rng = random.Random(seed * 7_000_003 + j)
    img = Image.open(out / src["path"]).convert("RGB")
    img2, ops = near_dup(img, rng)
    ext = rel.rsplit(".", 1)[1]
    buf = io.BytesIO()
    if ext == "jpg":
        img2.save(buf, "JPEG", quality=rng.randint(45, 80) if "reencode" in ops else rng.randint(80, 92))
    elif ext == "webp":
        img2.save(buf, "WEBP", quality=rng.randint(60, 85), method=2)
    else:
        img2.save(buf, "PNG", compress_level=1)
    (out / rel).parent.mkdir(parents=True, exist_ok=True)
    (out / rel).write_bytes(buf.getvalue())
    return {"path": rel, "category": src["category"], "text": src["text"], "width": img2.width,
            "height": img2.height, "dup_of": src["path"], "dup_kind": "near", "transform": ops}


_TAKEN: set[str] = set()


def _unique(out: Path, folder: str, name: str, salt: int) -> str:
    """Pick a path not used yet (decided in the parent process → no races)."""
    rel = f"{folder}/{name}"
    if rel.lower() in _TAKEN or (out / rel).exists():
        stem, ext = name.rsplit(".", 1)
        rel = f"{folder}/{stem} ({salt}).{ext}"
    _TAKEN.add(rel.lower())
    return rel


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exact-frac", type=float, default=0.05)
    ap.add_argument("--near-frac", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--clean", action="store_true", help="delete --out first")
    a = ap.parse_args(argv)

    out: Path = a.out.resolve()
    if a.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    n_exact, n_near = int(round(a.n * a.exact_frac)), int(round(a.n * a.near_frac))
    n_base = a.n - n_exact - n_near
    kinds = list(CATEGORY_WEIGHTS)
    weights = [CATEGORY_WEIGHTS[k] for k in kinds]
    plan = []
    for i in range(n_base):
        kind = rng.choices(kinds, weights)[0]
        ext = pick_format(rng, kind)
        plan.append((i, a.seed, out, kind, _unique(out, rng.choice(FOLDERS), pick_name(rng, kind, ext), i)))

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        base = list(ex.map(_make_base, plan, chunksize=16))
    print(f"generated {len(base)} base images in {time.time() - t0:.1f}s", file=sys.stderr)

    # exact duplicates: byte copies under a new name / folder
    records = list(base)
    pairs = []
    for j in range(n_exact):
        src = rng.choice(base)
        ext = src["path"].rsplit(".", 1)[1]
        rel = _unique(out, rng.choice(FOLDERS), rng.choice([
            f"Copy of {Path(src['path']).name}", f"{Path(src['path']).stem} (1).{ext}",
            pick_name(rng, src["category"], ext)]), 800_000 + j)
        if rel == src["path"]:
            rel = f"duplicates/{800_000 + j}_{Path(src['path']).name}"
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out / src["path"], out / rel)
        records.append({**src, "path": rel, "dup_of": src["path"], "dup_kind": "exact"})
        pairs.append({"a": src["path"], "b": rel, "kind": "exact"})

    near_srcs = []
    for j in range(n_near):
        src = rng.choice(base)
        ext = rng.choice(["jpg", "jpg", "png", "webp"])
        near_srcs.append((j, a.seed, out, src, _unique(out, rng.choice(FOLDERS), pick_name(rng, src["category"], ext), 900_000 + j)))
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        near = list(ex.map(_make_near, near_srcs, chunksize=8))
    for r in near:
        records.append(r)
        pairs.append({"a": r["dup_of"], "b": r["path"], "kind": "near", "transform": r["transform"]})

    counts: dict[str, int] = {}
    for r in records:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    gt = {
        "seed": a.seed, "n": len(records), "categories": counts,
        "files": {r["path"]: {k: v for k, v in r.items() if k != "path"} for r in records},
        "duplicate_pairs": pairs,
    }
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=1))
    n_files = sum(1 for p in out.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
    print(json.dumps({"out": str(out), "files": n_files, "records": len(records), "exact_pairs": n_exact,
                      "near_pairs": n_near, "categories": counts, "seconds": round(time.time() - t0, 1)}, indent=1))


if __name__ == "__main__":
    main()

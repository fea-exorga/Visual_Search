import React, { useCallback, useEffect, useMemo, useRef } from 'react';
import { Maximize2, Minus, Plus } from 'lucide-react';
import { thumbnailUrl } from '../../hooks/useApi';
import { easeInOutCubic, type GalaxyRendererProps } from './galaxyTypes';

const GRID = 96; // spatial hash resolution
const THUMB_ZOOM = 4.5; // zoom factor (relative to fit) at which thumbnails appear
const MAX_THUMBS = 200;
const CACHE_CAP = 500;
/** Screen area covered by overlay UI (toolbar on top, legend on the right). */
const INSET = { top: 64, right: 260, bottom: 24, left: 24 };

interface View {
  cx: number;
  cy: number;
  scale: number; // CSS px per world unit
}

interface Packed {
  n: number;
  xs: Float32Array;
  ys: Float32Array;
  ids: Int32Array;
  cl: Int32Array;
  groups: Map<number, number[]>; // cluster -> point indices
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  grid: Int32Array[]; // cell -> point indices
  cell: number;
  indexOf: Map<number, number>;
}

function pack(points: GalaxyRendererProps['data']['points']): Packed {
  const n = points.length;
  const xs = new Float32Array(n);
  const ys = new Float32Array(n);
  const ids = new Int32Array(n);
  const cl = new Int32Array(n);
  const groups = new Map<number, number[]>();
  const indexOf = new Map<number, number>();
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  points.forEach((p, i) => {
    xs[i] = p.x;
    ys[i] = p.y;
    ids[i] = p.image_id;
    cl[i] = p.cluster_id ?? -1;
    indexOf.set(p.image_id, i);
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
    let g = groups.get(cl[i]);
    if (!g) groups.set(cl[i], (g = []));
    g.push(i);
  });
  if (!n) {
    minX = minY = -1;
    maxX = maxY = 1;
  }
  const span = Math.max(maxX - minX, maxY - minY, 1e-6);
  const cell = span / GRID;
  const buckets: number[][] = Array.from({ length: GRID * GRID }, () => []);
  for (let i = 0; i < n; i++) {
    const gx = Math.min(GRID - 1, Math.floor((xs[i] - minX) / cell));
    const gy = Math.min(GRID - 1, Math.floor((ys[i] - minY) / cell));
    buckets[gy * GRID + gx].push(i);
  }
  return { n, xs, ys, ids, cl, groups, minX, maxX, minY, maxY, grid: buckets.map((b) => Int32Array.from(b)), cell, indexOf };
}

function makeStars(w: number, h: number, dpr: number): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = Math.max(1, w * dpr);
  c.height = Math.max(1, h * dpr);
  const g = c.getContext('2d')!;
  const grd = g.createRadialGradient(c.width * 0.5, c.height * 0.5, 0, c.width * 0.5, c.height * 0.5, Math.max(c.width, c.height) * 0.7);
  grd.addColorStop(0, '#0b1030');
  grd.addColorStop(0.55, '#050818');
  grd.addColorStop(1, '#020617');
  g.fillStyle = grd;
  g.fillRect(0, 0, c.width, c.height);
  let seed = 1337;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const count = Math.round((w * h) / 2200);
  for (let i = 0; i < count; i++) {
    const a = 0.15 + rnd() * 0.5;
    g.fillStyle = `rgba(203,213,225,${a})`;
    const s = (rnd() < 0.08 ? 1.6 : 0.8) * dpr;
    g.fillRect(rnd() * c.width, rnd() * c.height, s, s);
  }
  return c;
}

export default function Galaxy2D(props: GalaxyRendererProps) {
  const { data, onHover, onSelect } = props;
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const propsRef = useRef(props);
  propsRef.current = props;

  const packed = useMemo(() => pack(data.points), [data.points]);
  const packedRef = useRef(packed);
  packedRef.current = packed;

  const viewRef = useRef<View>({ cx: 0, cy: 0, scale: 100 });
  const fitScaleRef = useRef(100);
  const sizeRef = useRef({ w: 0, h: 0, dpr: 1 });
  const starsRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number>(0);
  const tweenRef = useRef<{ from: View; to: View; t0: number; dur: number } | null>(null);
  const hoverRef = useRef<number>(-1);
  const focusT0Ref = useRef<number>(0);
  const thumbsRef = useRef(new Map<number, HTMLImageElement>());
  const thumbDrawnRef = useRef<{ idx: number; size: number }[]>([]);
  const fittedRef = useRef(false);
  const interactingRef = useRef(false);
  const idleTimerRef = useRef<number>();

  // ---------- drawing ----------
  const draw = useCallback(() => {
    rafRef.current = 0;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    const { w, h, dpr } = sizeRef.current;
    if (!w || !h) return;
    const P = packedRef.current;
    const { clusterColors, highlight, activeClusters, focusId } = propsRef.current;
    const now = performance.now();
    let animating = false;

    // tween
    const tw = tweenRef.current;
    if (tw) {
      const t = Math.min(1, (now - tw.t0) / tw.dur);
      const e = easeInOutCubic(t);
      const ls = Math.log(tw.from.scale) + (Math.log(tw.to.scale) - Math.log(tw.from.scale)) * e;
      viewRef.current = {
        cx: tw.from.cx + (tw.to.cx - tw.from.cx) * e,
        cy: tw.from.cy + (tw.to.cy - tw.from.cy) * e,
        scale: Math.exp(ls),
      };
      if (t >= 1) tweenRef.current = null;
      else animating = true;
    }

    const { cx, cy, scale } = viewRef.current;
    const zoom = scale / fitScaleRef.current;
    const hw = w / 2, hh = h / 2;
    const sxOf = (i: number) => hw + (P.xs[i] - cx) * scale;
    const syOf = (i: number) => hh - (P.ys[i] - cy) * scale;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (starsRef.current) ctx.drawImage(starsRef.current, 0, 0, w, h);
    else {
      ctx.fillStyle = '#020617';
      ctx.fillRect(0, 0, w, h);
    }

    const base = P.n < 400 ? 3 : P.n < 3000 ? 2.2 : 1.6;
    const r = Math.max(1.4, Math.min(6, base * Math.sqrt(zoom)));
    const clusterFilter = activeClusters.size > 0;
    const pad = 100;
    const visible = (sx: number, sy: number) => sx > -pad && sx < w + pad && sy > -pad && sy < h + pad;

    // soft glow pass under the points (cheap: larger, very transparent)
    ctx.globalCompositeOperation = 'lighter';
    for (const [cid, idxs] of P.groups) {
      if (P.n > 2500 || interactingRef.current) break; // glow is a luxury for small sets / idle frames
      if (clusterFilter && !activeClusters.has(cid)) continue;
      if (highlight) continue;
      ctx.fillStyle = clusterColors.get(cid) ?? '#94a3b8';
      ctx.globalAlpha = 0.07;
      ctx.beginPath();
      const gr = r * 3.2;
      for (const i of idxs) {
        const sx = sxOf(i), sy = syOf(i);
        if (!visible(sx, sy)) continue;
        ctx.moveTo(sx + gr, sy);
        ctx.arc(sx, sy, gr, 0, Math.PI * 2);
      }
      ctx.fill();
    }
    ctx.globalCompositeOperation = 'source-over';

    // points, batched per cluster colour
    for (const [cid, idxs] of P.groups) {
      const dimCluster = clusterFilter && !activeClusters.has(cid);
      ctx.fillStyle = clusterColors.get(cid) ?? '#94a3b8';
      ctx.globalAlpha = dimCluster ? 0.07 : highlight ? 0.16 : 0.9;
      if (r < 2.3 && P.n > 2500) {
        // tiny points: squares are ~5x cheaper than arcs and indistinguishable
        const d = r * 1.8;
        for (const i of idxs) {
          const sx = sxOf(i), sy = syOf(i);
          if (!visible(sx, sy)) continue;
          ctx.fillRect(sx - d / 2, sy - d / 2, d, d);
        }
      } else {
        ctx.beginPath();
        for (const i of idxs) {
          const sx = sxOf(i), sy = syOf(i);
          if (!visible(sx, sy)) continue;
          ctx.moveTo(sx + r, sy);
          ctx.arc(sx, sy, r, 0, Math.PI * 2);
        }
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;

    // thumbnails when zoomed in
    const thumbDrawn: { idx: number; size: number }[] = [];
    if (zoom >= THUMB_ZOOM) {
      const size = Math.max(26, Math.min(96, 7 * zoom));
      const cand: { i: number; d: number }[] = [];
      for (let i = 0; i < P.n; i++) {
        const sx = sxOf(i), sy = syOf(i);
        if (sx < -size || sx > w + size || sy < -size || sy > h + size) continue;
        if (clusterFilter && !activeClusters.has(P.cl[i])) continue;
        const isHit = highlight ? highlight.has(P.ids[i]) : false;
        if (highlight && !isHit) continue;
        const d = (sx - hw) ** 2 + (sy - hh) ** 2;
        cand.push({ i, d });
      }
      if (cand.length <= MAX_THUMBS * 4) {
        cand.sort((a, b) => a.d - b.d);
        const chosen = cand.slice(0, MAX_THUMBS);
        // draw far-to-near so the centre ones are on top
        for (let k = chosen.length - 1; k >= 0; k--) {
          const i = chosen[k].i;
          const id = P.ids[i];
          const cache = thumbsRef.current;
          let img = cache.get(id);
          if (!img) {
            img = new Image();
            img.decoding = 'async';
            img.onload = () => requestDraw();
            img.src = thumbnailUrl(id);
            cache.set(id, img);
            if (cache.size > CACHE_CAP) {
              const first = cache.keys().next().value as number;
              cache.delete(first);
            }
          } else {
            cache.delete(id);
            cache.set(id, img); // LRU touch
          }
          const sx = sxOf(i), sy = syOf(i);
          const col = clusterColors.get(P.cl[i]) ?? '#94a3b8';
          if (img.complete && img.naturalWidth) {
            const s = Math.min(img.naturalWidth, img.naturalHeight);
            ctx.drawImage(img, (img.naturalWidth - s) / 2, (img.naturalHeight - s) / 2, s, s, sx - size / 2, sy - size / 2, size, size);
          } else {
            ctx.fillStyle = '#0f172a';
            ctx.fillRect(sx - size / 2, sy - size / 2, size, size);
          }
          ctx.strokeStyle = col;
          ctx.lineWidth = 1.5;
          ctx.strokeRect(sx - size / 2, sy - size / 2, size, size);
          thumbDrawn.push({ idx: i, size });
        }
      }
    }
    thumbDrawnRef.current = thumbDrawn;

    // search hits on top
    if (highlight && highlight.size) {
      ctx.save();
      ctx.shadowBlur = 12;
      for (const id of highlight) {
        const i = P.indexOf.get(id);
        if (i === undefined) continue;
        if (thumbDrawn.length && thumbDrawn.some((t) => t.idx === i)) continue;
        const sx = sxOf(i), sy = syOf(i);
        if (!visible(sx, sy)) continue;
        const col = clusterColors.get(P.cl[i]) ?? '#e2e8f0';
        ctx.shadowColor = col;
        ctx.fillStyle = col;
        ctx.beginPath();
        ctx.arc(sx, sy, r + 2.2, 0, Math.PI * 2);
        ctx.fill();
        ctx.lineWidth = 1.2;
        ctx.strokeStyle = 'rgba(255,255,255,0.9)';
        ctx.stroke();
      }
      ctx.restore();
    }

    // hovered point
    const hv = hoverRef.current;
    if (hv >= 0 && hv < P.n) {
      const sx = sxOf(hv), sy = syOf(hv);
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(sx, sy, r + 5, 0, Math.PI * 2);
      ctx.stroke();
    }

    // focused point (pulsing ring)
    if (focusId != null) {
      const i = P.indexOf.get(focusId);
      if (i !== undefined) {
        const sx = sxOf(i), sy = syOf(i);
        const t = ((now - focusT0Ref.current) % 1400) / 1400;
        ctx.strokeStyle = `rgba(165,180,252,${1 - t})`;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(sx, sy, r + 6 + t * 18, 0, Math.PI * 2);
        ctx.stroke();
        ctx.strokeStyle = '#e0e7ff';
        ctx.beginPath();
        ctx.arc(sx, sy, r + 4, 0, Math.PI * 2);
        ctx.stroke();
        if (now - focusT0Ref.current < 6000) animating = true;
      }
    }

    if (animating) rafRef.current = requestAnimationFrame(draw);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const requestDraw = useCallback(() => {
    if (!rafRef.current) rafRef.current = requestAnimationFrame(draw);
  }, [draw]);

  /** Mark the view as moving: cheaper frames until 150 ms after the last pan/zoom. */
  const markInteracting = useCallback(() => {
    interactingRef.current = true;
    window.clearTimeout(idleTimerRef.current);
    idleTimerRef.current = window.setTimeout(() => {
      interactingRef.current = false;
      requestDraw();
    }, 150);
  }, [requestDraw]);

  // ---------- fit / fly ----------
  /** Centre (cx, cy) so that world point (x, y) lands in the middle of the un-obscured area. */
  const centreOn = useCallback((x: number, y: number, scale: number): View => {
    const { w, h } = sizeRef.current;
    const ins = w > 700 ? INSET : { top: 64, right: 12, bottom: 12, left: 12 };
    const bx = (ins.left + (w - ins.right)) / 2 - w / 2;
    const by = (ins.top + (h - ins.bottom)) / 2 - h / 2;
    return { cx: x - bx / scale, cy: y + by / scale, scale };
  }, []);

  const fitView = useCallback((): View => {
    const P = packedRef.current;
    const { w, h } = sizeRef.current;
    const ins = w > 700 ? INSET : { top: 64, right: 12, bottom: 12, left: 12 };
    const spanX = Math.max(P.maxX - P.minX, 1e-3);
    const spanY = Math.max(P.maxY - P.minY, 1e-3);
    const scale = Math.min(((w - ins.left - ins.right) * 0.9) / spanX, ((h - ins.top - ins.bottom) * 0.9) / spanY);
    fitScaleRef.current = scale;
    return centreOn((P.minX + P.maxX) / 2, (P.minY + P.maxY) / 2, scale);
  }, [centreOn]);

  const animateTo = useCallback(
    (to: View, dur = 700) => {
      tweenRef.current = { from: { ...viewRef.current }, to, t0: performance.now(), dur };
      requestDraw();
    },
    [requestDraw],
  );

  // ---------- size ----------
  useEffect(() => {
    const el = containerRef.current!;
    const ro = new ResizeObserver(() => {
      const rect = el.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.round(rect.width), h = Math.round(rect.height);
      sizeRef.current = { w, h, dpr };
      const c = canvasRef.current!;
      c.width = Math.max(1, w * dpr);
      c.height = Math.max(1, h * dpr);
      starsRef.current = makeStars(w, h, dpr);
      const fit = fitView();
      if (!fittedRef.current && w > 0) {
        viewRef.current = fit;
        fittedRef.current = true;
      }
      requestDraw();
    });
    ro.observe(el);
    return () => {
      ro.disconnect();
      cancelAnimationFrame(rafRef.current);
      rafRef.current = 0;
    };
  }, [fitView, requestDraw]);

  // refit on new data
  useEffect(() => {
    if (sizeRef.current.w) viewRef.current = fitView();
    requestDraw();
  }, [packed, fitView, requestDraw]);

  // redraw on visual prop changes
  useEffect(() => {
    requestDraw();
  }, [props.highlight, props.activeClusters, props.clusterColors, requestDraw]);

  useEffect(() => {
    focusT0Ref.current = performance.now();
    requestDraw();
  }, [props.focusId, requestDraw]);

  // fly requests
  useEffect(() => {
    const f = props.fly;
    if (!f || !sizeRef.current.w) return;
    const { w, h } = sizeRef.current;
    const fitScale = fitScaleRef.current;
    const target = (Math.min(w - INSET.right, h - INSET.top) * 0.42) / Math.max(f.spread, 0.01);
    animateTo(centreOn(f.x, f.y, Math.max(fitScale * 0.6, Math.min(fitScale * 40, target))));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.fly?.nonce]);

  // ---------- picking ----------
  const pick = useCallback((mx: number, my: number): number => {
    const P = packedRef.current;
    const { cx, cy, scale } = viewRef.current;
    const { w, h } = sizeRef.current;
    const { activeClusters } = propsRef.current;
    // thumbnails first (they are big)
    const thumbs = thumbDrawnRef.current;
    for (let k = thumbs.length - 1; k >= 0; k--) {
      const { idx, size } = thumbs[k];
      const sx = w / 2 + (P.xs[idx] - cx) * scale;
      const sy = h / 2 - (P.ys[idx] - cy) * scale;
      if (Math.abs(mx - sx) <= size / 2 && Math.abs(my - sy) <= size / 2) return idx;
    }
    const wx = cx + (mx - w / 2) / scale;
    const wy = cy - (my - h / 2) / scale;
    const rad = 9 / scale;
    const g0x = Math.max(0, Math.floor((wx - rad - P.minX) / P.cell));
    const g1x = Math.min(GRID - 1, Math.floor((wx + rad - P.minX) / P.cell));
    const g0y = Math.max(0, Math.floor((wy - rad - P.minY) / P.cell));
    const g1y = Math.min(GRID - 1, Math.floor((wy + rad - P.minY) / P.cell));
    let best = -1;
    let bestD = rad * rad;
    for (let gy = g0y; gy <= g1y; gy++) {
      for (let gx = g0x; gx <= g1x; gx++) {
        const cellPts = P.grid[gy * GRID + gx];
        for (let k = 0; k < cellPts.length; k++) {
          const i = cellPts[k];
          if (activeClusters.size && !activeClusters.has(P.cl[i])) continue;
          const d = (P.xs[i] - wx) ** 2 + (P.ys[i] - wy) ** 2;
          if (d < bestD) {
            bestD = d;
            best = i;
          }
        }
      }
    }
    return best;
  }, []);

  const setHover = useCallback(
    (i: number) => {
      if (hoverRef.current === i) return;
      hoverRef.current = i;
      requestDraw();
      if (i < 0) {
        onHover(null);
        return;
      }
      const P = packedRef.current;
      const { cx, cy, scale } = viewRef.current;
      const { w, h } = sizeRef.current;
      onHover({
        imageId: P.ids[i],
        clusterId: P.cl[i],
        sx: w / 2 + (P.xs[i] - cx) * scale,
        sy: h / 2 - (P.ys[i] - cy) * scale,
      });
    },
    [onHover, requestDraw],
  );

  // ---------- interaction ----------
  const dragRef = useRef<{ id: number; x: number; y: number; moved: boolean } | null>(null);

  const local = (e: { clientX: number; clientY: number }) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  const onPointerDown = (e: React.PointerEvent) => {
    canvasRef.current!.setPointerCapture(e.pointerId);
    const p = local(e);
    dragRef.current = { id: e.pointerId, x: p.x, y: p.y, moved: false };
    tweenRef.current = null;
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const p = local(e);
    const d = dragRef.current;
    if (d && d.id === e.pointerId) {
      const dx = p.x - d.x, dy = p.y - d.y;
      if (!d.moved && Math.hypot(dx, dy) > 3) {
        d.moved = true;
        setHover(-1);
      }
      if (d.moved) {
        const v = viewRef.current;
        viewRef.current = { ...v, cx: v.cx - dx / v.scale, cy: v.cy + dy / v.scale };
        markInteracting();
        d.x = p.x;
        d.y = p.y;
        requestDraw();
      }
      return;
    }
    setHover(pick(p.x, p.y));
  };

  const onPointerUp = (e: React.PointerEvent) => {
    const d = dragRef.current;
    dragRef.current = null;
    if (!d || d.moved) return;
    const p = local(e);
    const i = pick(p.x, p.y);
    if (i >= 0) onSelect(packedRef.current.ids[i]);
  };

  const zoomAt = useCallback(
    (mx: number, my: number, factor: number) => {
      const v = viewRef.current;
      const { w, h } = sizeRef.current;
      const fit = fitScaleRef.current;
      const scale = Math.max(fit * 0.3, Math.min(fit * 80, v.scale * factor));
      const wx = v.cx + (mx - w / 2) / v.scale;
      const wy = v.cy - (my - h / 2) / v.scale;
      viewRef.current = { scale, cx: wx - (mx - w / 2) / scale, cy: wy + (my - h / 2) / scale };
      markInteracting();
      requestDraw();
    },
    [requestDraw, markInteracting],
  );

  useEffect(() => {
    const c = canvasRef.current!;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      tweenRef.current = null;
      const rect = c.getBoundingClientRect();
      const delta = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
      zoomAt(e.clientX - rect.left, e.clientY - rect.top, Math.exp(-delta * 0.0015));
      setHover(-1);
    };
    c.addEventListener('wheel', onWheel, { passive: false });
    return () => c.removeEventListener('wheel', onWheel);
  }, [zoomAt, setHover]);

  const onKeyDown = (e: React.KeyboardEvent) => {
    const { w, h } = sizeRef.current;
    const v = viewRef.current;
    const step = 60 / v.scale;
    let handled = true;
    switch (e.key) {
      case 'ArrowLeft': viewRef.current = { ...v, cx: v.cx - step }; break;
      case 'ArrowRight': viewRef.current = { ...v, cx: v.cx + step }; break;
      case 'ArrowUp': viewRef.current = { ...v, cy: v.cy + step }; break;
      case 'ArrowDown': viewRef.current = { ...v, cy: v.cy - step }; break;
      case '+': case '=': zoomAt(w / 2, h / 2, 1.3); break;
      case '-': case '_': zoomAt(w / 2, h / 2, 1 / 1.3); break;
      case '0': animateTo(fitView()); break;
      default: handled = false;
    }
    if (handled) {
      e.preventDefault();
      requestDraw();
    }
  };

  return (
    <div ref={containerRef} className="absolute inset-0 overflow-hidden">
      <canvas
        ref={canvasRef}
        tabIndex={0}
        aria-label="2D galaxy of images. Drag to pan, scroll to zoom, arrow keys to pan, plus and minus to zoom, 0 to reset."
        className="block h-full w-full cursor-grab touch-none outline-none active:cursor-grabbing"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={() => (dragRef.current = null)}
        onPointerLeave={() => !dragRef.current && setHover(-1)}
        onKeyDown={onKeyDown}
      />
      <div className="absolute bottom-4 left-4 flex flex-col overflow-hidden rounded-xl border border-slate-700/70 bg-slate-900/80 backdrop-blur">
        <button className="p-2 text-slate-300 hover:bg-slate-800" title="Zoom in (+)" onClick={() => zoomAt(sizeRef.current.w / 2, sizeRef.current.h / 2, 1.4)}>
          <Plus className="h-4 w-4" />
        </button>
        <button className="p-2 text-slate-300 hover:bg-slate-800" title="Zoom out (−)" onClick={() => zoomAt(sizeRef.current.w / 2, sizeRef.current.h / 2, 1 / 1.4)}>
          <Minus className="h-4 w-4" />
        </button>
        <button className="border-t border-slate-700/70 p-2 text-slate-300 hover:bg-slate-800" title="Fit (0)" onClick={() => animateTo(fitView())}>
          <Maximize2 className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}

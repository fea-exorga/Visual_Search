import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Eraser, Paintbrush, Undo2, Trash2 } from 'lucide-react';
import { rgbToHex } from '../../lib/format';

const SIZE = 256;
const DEBOUNCE_MS = 400;
const MAX_UNDO = 40;

const PALETTE = [
  '#111827', '#EF4444', '#F97316', '#FACC15', '#22C55E', '#14B8A6',
  '#3B82F6', '#6366F1', '#A855F7', '#EC4899', '#92400E', '#9CA3AF',
];

const BRUSHES = [
  { size: 4, label: 'Fine' },
  { size: 10, label: 'Medium' },
  { size: 22, label: 'Bold' },
];

interface SketchPadProps {
  initialPng: string | null;
  includeColors: boolean;
  extractedColors: string[];
  onToggleIncludeColors: (v: boolean) => void;
  onChange: (png: string | null, colors: string[]) => void;
  busy?: boolean;
}

/** Dominant stroke colours (ignores the white background). */
function extractStrokeColors(ctx: CanvasRenderingContext2D): string[] {
  const { data } = ctx.getImageData(0, 0, SIZE, SIZE);
  const buckets = new Map<number, { r: number; g: number; b: number; n: number }>();
  let inked = 0;
  for (let i = 0; i < data.length; i += 8) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    if (r > 232 && g > 232 && b > 232) continue;
    inked++;
    const key = ((r >> 5) << 6) | ((g >> 5) << 3) | (b >> 5);
    const bk = buckets.get(key);
    if (bk) {
      bk.r += r; bk.g += g; bk.b += b; bk.n++;
    } else buckets.set(key, { r, g, b, n: 1 });
  }
  if (!inked) return [];
  return [...buckets.values()]
    .filter((b) => b.n / inked > 0.06)
    .sort((a, b) => b.n - a.n)
    .slice(0, 3)
    .map((b) => rgbToHex(b.r / b.n, b.g / b.n, b.b / b.n));
}

function isBlank(ctx: CanvasRenderingContext2D): boolean {
  const { data } = ctx.getImageData(0, 0, SIZE, SIZE);
  for (let i = 0; i < data.length; i += 16) {
    if (data[i] < 245 || data[i + 1] < 245 || data[i + 2] < 245) return false;
  }
  return true;
}

export default function SketchPad({
  initialPng,
  includeColors,
  extractedColors,
  onToggleIncludeColors,
  onChange,
  busy,
}: SketchPadProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);
  const undoRef = useRef<ImageData[]>([]);
  const drawingRef = useRef<{ id: number; x: number; y: number } | null>(null);
  const timerRef = useRef<number>();
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;

  const [color, setColor] = useState(PALETTE[6]);
  const [brush, setBrush] = useState(BRUSHES[1].size);
  const [eraser, setEraser] = useState(false);
  const [undoCount, setUndoCount] = useState(0);
  const [empty, setEmpty] = useState(!initialPng);

  // init canvas (white bg) and restore a previous sketch if any
  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    ctxRef.current = ctx;
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, SIZE, SIZE);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    if (initialPng) {
      const img = new Image();
      img.onload = () => ctx.drawImage(img, 0, 0, SIZE, SIZE);
      img.src = initialPng;
    }
    return () => window.clearTimeout(timerRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const emit = useCallback((immediate = false) => {
    window.clearTimeout(timerRef.current);
    const run = () => {
      const ctx = ctxRef.current;
      if (!ctx) return;
      if (isBlank(ctx)) {
        setEmpty(true);
        onChangeRef.current(null, []);
      } else {
        setEmpty(false);
        onChangeRef.current(canvasRef.current!.toDataURL('image/png'), extractStrokeColors(ctx));
      }
    };
    if (immediate) run();
    else timerRef.current = window.setTimeout(run, DEBOUNCE_MS);
  }, []);

  const pushUndo = () => {
    const ctx = ctxRef.current!;
    undoRef.current.push(ctx.getImageData(0, 0, SIZE, SIZE));
    if (undoRef.current.length > MAX_UNDO) undoRef.current.shift();
    setUndoCount(undoRef.current.length);
  };

  const toCanvas = (e: { clientX: number; clientY: number }) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * SIZE,
      y: ((e.clientY - rect.top) / rect.height) * SIZE,
    };
  };

  const widthFor = (e: React.PointerEvent | PointerEvent) => {
    const p = e.pointerType === 'pen' && e.pressure > 0 ? 0.35 + e.pressure * 1.1 : 1;
    return (eraser ? brush * 1.8 : brush) * p;
  };

  const strokeTo = (x: number, y: number, w: number) => {
    const ctx = ctxRef.current!;
    const last = drawingRef.current!;
    ctx.strokeStyle = eraser ? '#FFFFFF' : color;
    ctx.lineWidth = w;
    ctx.beginPath();
    ctx.moveTo(last.x, last.y);
    ctx.lineTo(x, y);
    ctx.stroke();
    last.x = x;
    last.y = y;
  };

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (e.button !== 0 && e.pointerType === 'mouse') return;
    e.preventDefault();
    canvasRef.current!.setPointerCapture(e.pointerId);
    window.clearTimeout(timerRef.current);
    pushUndo();
    const { x, y } = toCanvas(e);
    drawingRef.current = { id: e.pointerId, x, y };
    const ctx = ctxRef.current!;
    ctx.fillStyle = eraser ? '#FFFFFF' : color;
    ctx.beginPath();
    ctx.arc(x, y, widthFor(e) / 2, 0, Math.PI * 2);
    ctx.fill();
    setEmpty(false);
  };

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const d = drawingRef.current;
    if (!d || d.id !== e.pointerId) return;
    const native = e.nativeEvent;
    const events = typeof native.getCoalescedEvents === 'function' ? native.getCoalescedEvents() : [];
    if (events.length) {
      for (const ce of events) {
        const p = toCanvas(ce);
        strokeTo(p.x, p.y, widthFor(ce));
      }
    } else {
      const p = toCanvas(e);
      strokeTo(p.x, p.y, widthFor(e));
    }
  };

  const endStroke = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const d = drawingRef.current;
    if (!d || d.id !== e.pointerId) return;
    drawingRef.current = null;
    emit();
  };

  const undo = useCallback(() => {
    const snap = undoRef.current.pop();
    if (!snap) return;
    ctxRef.current!.putImageData(snap, 0, 0);
    setUndoCount(undoRef.current.length);
    emit();
  }, [emit]);

  const clear = () => {
    if (empty) return;
    pushUndo();
    const ctx = ctxRef.current!;
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, SIZE, SIZE);
    emit(true);
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
      e.preventDefault();
      undo();
    } else if (e.key === 'e') setEraser(true);
    else if (e.key === 'b') setEraser(false);
    else if (e.key === '[') setBrush((b) => Math.max(2, b - 3));
    else if (e.key === ']') setBrush((b) => Math.min(40, b + 3));
  };

  return (
    <div className="flex flex-col gap-3" onKeyDown={onKeyDown}>
      <div className="relative">
        <canvas
          ref={canvasRef}
          width={SIZE}
          height={SIZE}
          tabIndex={0}
          aria-label="Sketch canvas. Draw to search. Ctrl+Z undo, E eraser, B brush, [ and ] change size."
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={endStroke}
          onPointerCancel={endStroke}
          onContextMenu={(e) => e.preventDefault()}
          className={`block aspect-square w-full touch-none select-none rounded-xl bg-white shadow-inner ring-1 ring-slate-700 ${
            eraser ? 'cursor-cell' : 'cursor-crosshair'
          }`}
        />
        {empty && (
          <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center text-center text-xs text-slate-400">
            <Paintbrush className="mb-1 h-5 w-5 text-slate-300" />
            Draw here — results update
            <br /> as you sketch
          </div>
        )}
        {busy && (
          <span className="absolute right-2 top-2 h-2 w-2 animate-ping rounded-full bg-indigo-500" aria-hidden />
        )}
      </div>

      <div className="flex min-w-0 flex-1 flex-col gap-3">
        <div>
          <div className="field-label">Colour</div>
          <div className="flex flex-wrap items-center gap-1.5">
            {PALETTE.map((c) => (
              <button
                key={c}
                onClick={() => {
                  setColor(c);
                  setEraser(false);
                }}
                aria-label={`Brush colour ${c}`}
                aria-pressed={!eraser && color === c}
                className={`h-6 w-6 rounded-full ring-offset-2 ring-offset-slate-900 transition hover:scale-110 ${
                  !eraser && color === c ? 'ring-2 ring-indigo-300' : 'ring-1 ring-white/10'
                }`}
                style={{ backgroundColor: c }}
              />
            ))}
            <label className="relative h-6 w-6 overflow-hidden rounded-full ring-1 ring-white/20" title="Custom colour">
              <span
                className="absolute inset-0"
                style={{ background: 'conic-gradient(red, yellow, lime, cyan, blue, magenta, red)' }}
              />
              <input
                type="color"
                value={color}
                onChange={(e) => {
                  setColor(e.target.value.toUpperCase());
                  setEraser(false);
                }}
                className="absolute inset-0 h-full w-full opacity-0"
                aria-label="Custom brush colour"
              />
            </label>
          </div>
        </div>

        <div className="flex flex-wrap items-end gap-3">
          <div>
            <div className="field-label">Brush</div>
            <div className="flex items-center gap-1 rounded-lg border border-slate-700/80 bg-slate-950/60 p-0.5">
              {BRUSHES.map((b) => (
                <button
                  key={b.size}
                  onClick={() => setBrush(b.size)}
                  aria-pressed={brush === b.size}
                  title={`${b.label} brush`}
                  className={`flex h-7 w-8 items-center justify-center rounded-md transition ${
                    brush === b.size ? 'bg-slate-700' : 'hover:bg-slate-800'
                  }`}
                >
                  <span
                    className="rounded-full bg-slate-200"
                    style={{ width: Math.max(3, b.size / 1.6), height: Math.max(3, b.size / 1.6) }}
                  />
                </button>
              ))}
            </div>
          </div>
          <div className="flex gap-1">
            <button
              onClick={() => setEraser((v) => !v)}
              aria-pressed={eraser}
              className={`btn-secondary h-8 ${eraser ? '!border-indigo-500 !bg-indigo-500/20 text-indigo-200' : ''}`}
              title="Eraser (E)"
            >
              <Eraser className="h-3.5 w-3.5" />
            </button>
            <button onClick={undo} disabled={undoCount === 0} className="btn-secondary h-8" title="Undo (Ctrl+Z)">
              <Undo2 className="h-3.5 w-3.5" />
            </button>
            <button onClick={clear} disabled={empty} className="btn-secondary h-8" title="Clear">
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        <label className="flex cursor-pointer items-center gap-2 text-xs text-slate-300">
          <input
            type="checkbox"
            checked={includeColors}
            onChange={(e) => onToggleIncludeColors(e.target.checked)}
            className="h-3.5 w-3.5 rounded border-slate-600 bg-slate-800 accent-indigo-500"
          />
          Include sketch colours
          <span className="flex gap-0.5">
            {extractedColors.map((c) => (
              <span key={c} className="h-3 w-3 rounded-sm ring-1 ring-white/10" style={{ backgroundColor: c }} />
            ))}
          </span>
        </label>
      </div>
    </div>
  );
}

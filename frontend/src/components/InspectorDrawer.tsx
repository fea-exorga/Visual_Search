import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  X,
  Copy,
  Check,
  FileText,
  Palette,
  Hash,
  Box,
  AlertTriangle,
  Copy as CopyIcon,
  Orbit,
} from 'lucide-react';
import { getImageDetail, thumbnailUrl } from '../hooks/useApi';
import { formatBytes } from '../lib/format';
import { ErrorState } from './ui/States';
import type { ColorSwatch } from '../types';

interface InspectorDrawerProps {
  imageId: number | null;
  onClose: () => void;
  onFindSimilar: (imageId: number) => void;
  onShowInGalaxy: (imageId: number) => void;
}

function confidenceColor(conf: number): string {
  if (conf >= 0.85) return 'bg-emerald-900/60 text-emerald-400';
  if (conf >= 0.6) return 'bg-amber-900/60 text-amber-400';
  return 'bg-red-900/60 text-red-400';
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // fallback
    }
  };

  return (
    <button
      onClick={handleCopy}
      className="flex items-center gap-1 rounded-md bg-slate-700 px-2 py-1 text-[10px] text-slate-300 transition-colors hover:bg-slate-600"
    >
      {copied ? (
        <>
          <Check className="h-3 w-3 text-emerald-400" />
          Copied
        </>
      ) : (
        <>
          <Copy className="h-3 w-3" />
          Copy
        </>
      )}
    </button>
  );
}

function SwatchRow({ swatch, isSelected, onClick }: { swatch: ColorSwatch; isSelected: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`flex flex-col items-center gap-1 transition-transform duration-200 ${
        isSelected ? 'scale-110' : 'hover:scale-105'
      }`}
    >
      <div
        className={`h-10 w-10 rounded-lg border-2 transition-colors ${
          isSelected ? 'border-white shadow-lg' : 'border-transparent'
        }`}
        style={{ backgroundColor: swatch.hex_code }}
      />
      <span className="text-[10px] text-slate-400">{swatch.hex_code}</span>
      <span className="text-[9px] text-slate-500">
        {(swatch.weight * 100).toFixed(0)}%
      </span>
    </button>
  );
}

export default function InspectorDrawer({ imageId, onClose, onFindSimilar, onShowInGalaxy }: InspectorDrawerProps) {
  const [selectedSwatch, setSelectedSwatch] = useState<number | null>(null);

  const { data: detail, isLoading, isError, error } = useQuery({
    queryKey: ['imageDetail', imageId],
    queryFn: () => getImageDetail(imageId!),
    enabled: imageId !== null,
  });

  if (imageId === null) return null;

  const filename = detail?.file_path.split('/').pop() || '';

  return (
    <aside
      aria-label="Image inspector"
      className="absolute inset-y-0 right-0 z-30 flex w-[340px] flex-col border-l border-slate-700/60 bg-slate-900/95 shadow-2xl shadow-black/60 backdrop-blur-md 2xl:relative 2xl:z-auto 2xl:shrink-0 2xl:bg-slate-900/80 2xl:shadow-none"
    >
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-700/40 px-4 py-3">
        <h3 className="text-sm font-semibold text-slate-200">Inspector</h3>
        <button
          onClick={onClose}
          aria-label="Close inspector (Esc)"
          title="Close (Esc)"
          className="rounded-md p-1 text-slate-400 transition-colors hover:bg-slate-800 hover:text-slate-200"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      {isLoading ? (
        <div className="flex flex-1 items-center justify-center">
          <div className="h-6 w-6 animate-spin rounded-full border-2 border-slate-600 border-t-indigo-500" />
        </div>
      ) : detail ? (
        <div className="flex-1 overflow-y-auto">
          {/* Preview */}
          <div className="checker aspect-video w-full overflow-hidden bg-slate-950">
            <img
              src={thumbnailUrl(detail.image_id)}
              alt={filename}
              className="h-full w-full object-contain"
            />
          </div>

          {/* Actions */}
          <div className="grid grid-cols-2 gap-2 border-b border-slate-800 px-4 py-3">
            <button onClick={() => onFindSimilar(detail.image_id)} className="btn-primary px-2" disabled={detail.has_embedding === false}>
              <CopyIcon className="h-3.5 w-3.5" /> Find similar
            </button>
            <button onClick={() => onShowInGalaxy(detail.image_id)} className="btn-secondary px-2" disabled={detail.has_embedding === false}>
              <Orbit className="h-3.5 w-3.5" /> Show in galaxy
            </button>
          </div>

          {/* Section A — File & Hash Metadata */}
          <div className="border-b border-slate-800 px-4 py-3">
            <div className="mb-1 flex items-center gap-2">
              <Hash className="h-4 w-4 text-slate-500" />
              <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                File Metadata
              </h4>
            </div>
            <p className="truncate text-sm font-medium text-slate-200" title={filename}>{filename}</p>
            <div className="mt-1 flex items-start gap-2">
              <p className="min-w-0 flex-1 break-all font-mono text-[10px] leading-snug text-slate-500">{detail.file_path}</p>
              <CopyButton text={detail.file_path} />
            </div>
            <div className="mt-2 grid grid-cols-2 gap-y-1 text-xs">
              <span className="text-slate-500">Dimensions</span>
              <span className="text-slate-300">
                {detail.width} × {detail.height}
              </span>
              <span className="text-slate-500">File Size</span>
              <span className="text-slate-300">{formatBytes(detail.file_size)}</span>
              <span className="text-slate-500">Format</span>
              <span className="text-slate-300 uppercase">{detail.format}</span>
            </div>
            <div className="mt-2 flex items-center gap-2">
              <span className="text-[10px] font-mono text-slate-500">
                SHA-256: {detail.sha256?.slice(0, 16)}…
              </span>
              {detail.sha256 && <CopyButton text={detail.sha256} />}
            </div>
          </div>

          {/* Section B — Dominant Color Palette */}
          <div className="border-b border-slate-800 px-4 py-3">
            <div className="mb-2 flex items-center gap-2">
              <Palette className="h-4 w-4 text-slate-500" />
              <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                Color Palette
              </h4>
            </div>
            {detail.colors && detail.colors.length > 0 ? (
              <div className="flex justify-between gap-2">
                {detail.colors.map((c) => (
                  <SwatchRow
                    key={c.rank}
                    swatch={c}
                    isSelected={selectedSwatch === c.rank}
                    onClick={() =>
                      setSelectedSwatch(selectedSwatch === c.rank ? null : c.rank)
                    }
                  />
                ))}
              </div>
            ) : (
              <p className="text-xs italic text-slate-600">No colors extracted</p>
            )}
            {/* Percentage bars */}
            {detail.colors && detail.colors.length > 0 && (
              <div className="mt-3 space-y-1">
                {detail.colors.map((c) => (
                  <div key={c.rank} className="flex items-center gap-2">
                    <div
                      className="h-2 rounded-full"
                      style={{
                        backgroundColor: c.hex_code,
                        width: `${c.weight * 100}%`,
                        minWidth: '4px',
                      }}
                    />
                    <span className="text-[10px] text-slate-500 whitespace-nowrap">
                      rgb({c.r},{c.g},{c.b}) {(c.weight * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Section C — Extracted OCR Text */}
          <div className="border-b border-slate-800 px-4 py-3">
            <div className="mb-2 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FileText className="h-4 w-4 text-slate-500" />
                <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Extracted Text
                </h4>
              </div>
              {detail.ocr && detail.ocr.word_count > 0 && (
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-slate-500">
                    {detail.ocr.word_count} words
                  </span>
                  <span
                    className={`rounded-full px-2 py-0.5 text-[10px] font-medium ${confidenceColor(
                      detail.ocr.mean_confidence
                    )}`}
                  >
                    {(detail.ocr.mean_confidence * 100).toFixed(0)}% Confidence
                  </span>
                </div>
              )}
            </div>
            {detail.ocr && detail.ocr.raw_text ? (
              <div>
                <div className="max-h-48 overflow-y-auto rounded-lg bg-slate-800 p-3 font-mono text-xs text-slate-300">
                  {detail.ocr.raw_text}
                </div>
                <div className="mt-2 flex justify-end">
                  <CopyButton text={detail.ocr.raw_text} />
                </div>
              </div>
            ) : (
              <p className="text-xs italic text-slate-600">
                No text detected in this image
              </p>
            )}
          </div>

          {/* Section D — Vision Embedding Status */}
          <div className="px-4 py-3">
            <div className="mb-2 flex items-center gap-2">
              <Box className="h-4 w-4 text-slate-500" />
              <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                Vision Embedding
              </h4>
            </div>
            <div className="grid grid-cols-2 gap-y-1 text-xs">
              <span className="text-slate-500">Model</span>
              <span className="truncate text-slate-300">
                {detail.model_name || '—'}
              </span>
              <span className="text-slate-500">Dimensions</span>
              <span className="text-slate-300">
                {detail.embedding_dim != null ? `${detail.embedding_dim}-d float32` : '—'}
              </span>
              <span className="text-slate-500">L2 Norm</span>
              <span className="flex items-center gap-1 text-slate-300">
                {detail.l2_norm?.toFixed(3) ?? '—'}
                {detail.l2_norm != null && (
                  <Check className="h-3 w-3 text-emerald-500" />
                )}
              </span>
              <span className="text-slate-500">Image ID</span>
              <span className="text-slate-300">#{detail.image_id}</span>
            </div>
            <div className="mt-2">
              {detail.status === 'COMPLETED' && detail.has_embedding !== false ? (
                <span className="inline-flex items-center gap-1 rounded-full bg-emerald-900/60 px-2.5 py-0.5 text-xs font-medium text-emerald-400">
                  <Check className="h-3 w-3" />
                  Indexed
                </span>
              ) : (
                <span className="inline-flex items-center gap-1 rounded-full bg-amber-900/60 px-2.5 py-0.5 text-xs font-medium text-amber-400">
                  <AlertTriangle className="h-3 w-3" />
                  {detail.status === 'ERROR' ? 'Error' : detail.has_embedding === false ? 'No embedding' : 'Pending'}
                </span>
              )}
            </div>
          </div>
        </div>
      ) : isError ? (
        <ErrorState error={error} />
      ) : null}
    </aside>
  );
}

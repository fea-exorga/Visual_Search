import React from 'react';
import { FileText } from 'lucide-react';
import { thumbnailUrl } from '../../hooks/useApi';
import { basename } from '../../lib/format';
import type { SearchHit } from '../../types';

export const SCORE_COLORS = {
  semantic: '#818CF8',
  ocr: '#FBBF24',
  color: '#F472B6',
} as const;

const SCORE_LABELS: Record<keyof typeof SCORE_COLORS, string> = {
  semantic: 'Semantic',
  ocr: 'OCR',
  color: 'Colour',
};

interface ResultCardProps {
  hit: SearchHit;
  rank: number;
  isSelected: boolean;
  onClick: () => void;
  clusterLabel?: { label: string; color: string } | null;
}

function pct(v: number | null | undefined) {
  return Math.max(0, Math.min(1, v ?? 0)) * 100;
}

export default function ResultCard({ hit, rank, isSelected, onClick, clusterLabel }: ResultCardProps) {
  const name = basename(hit.file_path);
  const parts = (Object.keys(SCORE_COLORS) as (keyof typeof SCORE_COLORS)[]).filter(
    (k) => hit.scores?.[k] != null,
  );

  return (
    <button
      onClick={onClick}
      title={hit.file_path}
      className={`group relative flex flex-col overflow-hidden rounded-xl border bg-slate-900/80 text-left transition duration-200 hover:-translate-y-0.5 hover:shadow-xl hover:shadow-indigo-950/50 focus-visible:-translate-y-0.5 ${
        isSelected ? 'border-indigo-400 ring-2 ring-indigo-500/40' : 'border-slate-800 hover:border-slate-600'
      }`}
    >
      <div className="relative aspect-square w-full overflow-hidden bg-slate-950">
        <img
          src={hit.thumbnail_url || thumbnailUrl(hit.image_id)}
          alt={name}
          loading="lazy"
          className="h-full w-full object-cover transition duration-300 group-hover:scale-105"
        />
        <span className="absolute left-2 top-2 rounded-md bg-slate-950/70 px-1.5 py-0.5 font-mono text-[10px] text-slate-300 backdrop-blur">
          #{rank}
        </span>
        <span className="absolute right-2 top-2 rounded-md bg-slate-950/70 px-1.5 py-0.5 font-mono text-[10px] font-semibold text-indigo-200 backdrop-blur">
          {(hit.score * 100).toFixed(0)}
        </span>

        {/* Score breakdown — hover / keyboard focus */}
        {parts.length > 0 && (
          <div className="absolute inset-x-0 bottom-0 translate-y-full bg-gradient-to-t from-slate-950 via-slate-950/90 to-transparent px-2.5 pb-2 pt-6 opacity-0 transition duration-200 group-hover:translate-y-0 group-hover:opacity-100 group-focus-visible:translate-y-0 group-focus-visible:opacity-100">
            {parts.map((k) => (
              <div key={k} className="mb-1 last:mb-0">
                <div className="flex justify-between text-[10px] text-slate-300">
                  <span>{SCORE_LABELS[k]}</span>
                  <span className="font-mono">{pct(hit.scores[k]).toFixed(0)}%</span>
                </div>
                <div className="h-1 overflow-hidden rounded-full bg-slate-800">
                  <div className="h-full rounded-full" style={{ width: `${pct(hit.scores[k])}%`, backgroundColor: SCORE_COLORS[k] }} />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Total score bar, stacked by component */}
      <div className="flex h-1 w-full bg-slate-800" aria-hidden>
        {parts.length > 0 ? (
          parts.map((k) => (
            <div
              key={k}
              style={{ width: `${(pct(hit.score) * pct(hit.scores[k])) / Math.max(1, parts.reduce((s, p) => s + pct(hit.scores[p]), 0))}%`, backgroundColor: SCORE_COLORS[k] }}
            />
          ))
        ) : (
          <div className="bg-indigo-400" style={{ width: `${pct(hit.score)}%` }} />
        )}
      </div>

      {/* Palette strip */}
      {hit.colors?.length > 0 && (
        <div className="flex h-1.5 w-full">
          {hit.colors.map((c, i) => (
            <div key={i} style={{ backgroundColor: c.hex_code, width: `${(c.weight || 0.2) * 100}%` }} />
          ))}
        </div>
      )}

      <div className="flex min-h-[44px] flex-col gap-0.5 px-2.5 py-2">
        <div className="flex items-center gap-1.5">
          {clusterLabel && (
            <span
              className="h-1.5 w-1.5 shrink-0 rounded-full"
              style={{ backgroundColor: clusterLabel.color }}
              title={clusterLabel.label}
            />
          )}
          <p className="truncate text-xs text-slate-300">{name}</p>
        </div>
        {hit.ocr_snippet ? (
          <p className="flex items-start gap-1 text-[10px] leading-snug text-slate-500">
            <FileText className="mt-px h-2.5 w-2.5 shrink-0" />
            <span className="line-clamp-2">{hit.ocr_snippet}</span>
          </p>
        ) : null}
      </div>
    </button>
  );
}

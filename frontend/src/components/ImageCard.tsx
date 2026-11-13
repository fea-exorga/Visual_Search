import React from 'react';
import { thumbnailUrl } from '../hooks/useApi';
import type { ImageSummary } from '../types';

interface ImageCardProps {
  image: ImageSummary;
  onClick: () => void;
  isSelected: boolean;
}

function statusDotColor(status: string) {
  switch (status) {
    case 'COMPLETED':
      return 'bg-emerald-500';
    case 'ERROR':
      return 'bg-red-500';
    default:
      return 'bg-amber-500';
  }
}

function filenameFromPath(path: string): string {
  return path.split('/').pop() || path;
}

export default function ImageCard({ image, onClick, isSelected }: ImageCardProps) {
  const hasOcr = image.ocr_snippet && image.ocr_snippet.length > 0;
  const wordCount = hasOcr ? image.ocr_snippet!.split(/\s+/).length : 0;

  return (
    <button
      onClick={onClick}
      title={filenameFromPath(image.file_path)}
      className={`group relative flex flex-col overflow-hidden rounded-xl border transition duration-200 hover:-translate-y-0.5 hover:shadow-xl hover:shadow-indigo-950/50 ${
        isSelected
          ? 'border-indigo-400 ring-2 ring-indigo-500/40'
          : 'border-slate-800 hover:border-slate-600'
      } bg-slate-900/80`}
    >
      {/* Thumbnail */}
      <div className="relative aspect-square w-full overflow-hidden bg-slate-900">
        <img
          src={thumbnailUrl(image.image_id)}
          alt={filenameFromPath(image.file_path)}
          loading="lazy"
          className="h-full w-full object-cover transition-transform duration-200 group-hover:scale-105"
        />

        {/* Status dot top-left */}
        <span
          className={`absolute left-2 top-2 h-2.5 w-2.5 rounded-full ring-2 ring-slate-900/80 ${statusDotColor(image.status)}`}
        />

        {/* OCR badge top-right */}
        {hasOcr && (
          <span className="absolute right-2 top-2 rounded-full bg-indigo-600/80 px-2 py-0.5 text-[10px] font-medium text-white backdrop-blur-sm">
            OCR ✓ {wordCount}w
          </span>
        )}
      </div>

      {/* Color bar */}
      {image.colors && image.colors.length > 0 && (
        <div className="flex h-1.5 w-full">
          {image.colors.map((c, i) => (
            <div
              key={i}
              style={{
                backgroundColor: c.hex_code,
                width: `${c.weight * 100}%`,
              }}
            />
          ))}
        </div>
      )}

      {/* Filename label */}
      <div className="px-2.5 py-2">
        <p className="truncate text-left text-xs text-slate-400">
          {filenameFromPath(image.file_path)}
        </p>
      </div>
    </button>
  );
}

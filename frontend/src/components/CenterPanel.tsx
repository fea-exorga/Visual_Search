import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Play, Pause, RotateCcw, Zap, Clock, ImageIcon, FolderOpen, HardDrive } from 'lucide-react';
import { getStats } from '../hooks/useApi';
import type { IndexingState } from '../hooks/useIndexing';
import { formatBytes } from '../lib/format';
import ImageGrid from './ImageGrid';
import type { IndexProgress } from '../types';

interface CenterPanelProps {
  folderId: number | null;
  selectedImageId: number | null;
  onSelectImage: (imageId: number) => void;
  indexing: IndexingState;
}

function formatEta(sec: number | null | undefined): string {
  if (sec == null || sec <= 0) return '--:--';
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

function progressStages(p: IndexProgress | null) {
  const t = p?.total || 0;
  const pc = (v: number) => (t ? Math.round((v / t) * 100) : 0);
  return [
    { label: 'Scan', pct: pc(p?.scanned ?? 0) },
    { label: 'Palette', pct: pc(p?.color_done ?? 0) },
    { label: 'OCR', pct: pc(p?.ocr_done ?? 0) },
    { label: 'Embeddings', pct: pc(p?.embedded ?? 0) },
  ];
}

/** Library view: pipeline status banner + paginated grid of every indexed image. */
export default function CenterPanel({ folderId, selectedImageId, onSelectImage, indexing }: CenterPanelProps) {
  const { data: stats } = useQuery({ queryKey: ['stats'], queryFn: getStats, refetchInterval: 10000 });
  const { progress, isRunning } = indexing;
  // idle: show overall completion from /stats instead of an empty pipeline
  const idleProgress: IndexProgress | null =
    !progress && stats && stats.total_images
      ? (() => {
          const done = stats.by_status?.COMPLETED ?? 0;
          return { scanned: stats.total_images, color_done: done, ocr_done: done, embedded: done, skipped: 0, failed: 0, total: stats.total_images, stage: 'idle', eta_sec: null, images_per_sec: null };
        })()
      : null;
  const stages = progressStages(progress ?? idleProgress);

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-y-auto">
      <div className="sticky top-0 z-10 border-b border-slate-800/60 bg-slate-950/85 px-5 py-3 backdrop-blur-md">
        <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
          <div className="flex items-center gap-4 text-xs text-slate-400">
            <span className="flex items-center gap-1.5">
              <ImageIcon className="h-3.5 w-3.5 text-indigo-300" />
              <b className="text-slate-200">{stats?.total_images ?? 0}</b> images
            </span>
            <span className="flex items-center gap-1.5">
              <FolderOpen className="h-3.5 w-3.5 text-indigo-300" />
              <b className="text-slate-200">{stats?.total_folders ?? 0}</b> folders
            </span>
            <span className="flex items-center gap-1.5">
              <HardDrive className="h-3.5 w-3.5 text-indigo-300" />
              {formatBytes(stats?.disk_usage_bytes ?? 0)}
            </span>
            {stats?.by_status?.ERROR ? <span className="text-rose-400">{stats.by_status.ERROR} errors</span> : null}
          </div>

          <div className="flex min-w-[280px] flex-1 items-center gap-2">
            {stages.map((stage) => (
              <div key={stage.label} className="flex-1">
                <div className="mb-1 flex justify-between text-[10px] text-slate-500">
                  <span>{stage.label}</span>
                  <span className="font-mono">{stage.pct}%</span>
                </div>
                <div className="h-1.5 overflow-hidden rounded-full bg-slate-800">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-violet-400 transition-all duration-500"
                    style={{ width: `${stage.pct}%` }}
                  />
                </div>
              </div>
            ))}
          </div>

          <div className="flex items-center gap-3 text-xs text-slate-400">
            {isRunning && progress && (
              <>
                <span className="flex items-center gap-1">
                  <Zap className="h-3.5 w-3.5 text-amber-400" />
                  {progress.images_per_sec?.toFixed(1) ?? '—'} img/s
                </span>
                <span className="flex items-center gap-1">
                  <Clock className="h-3.5 w-3.5 text-sky-400" />~{formatEta(progress.eta_sec)}
                </span>
                {progress.failed > 0 && <span className="text-rose-400">{progress.failed} failed</span>}
              </>
            )}
            {!isRunning ? (
              <button onClick={() => indexing.start(false)} disabled={!indexing.canStart} className="btn-primary py-1.5">
                <Play className="h-3.5 w-3.5" /> Start
              </button>
            ) : (
              <button onClick={() => (indexing.isPaused ? indexing.resume() : indexing.pause())} className="btn-secondary">
                {indexing.isPaused ? <Play className="h-3.5 w-3.5" /> : <Pause className="h-3.5 w-3.5" />}
                {indexing.isPaused ? 'Resume' : 'Pause'}
              </button>
            )}
            <button onClick={() => indexing.start(true)} disabled={!indexing.canStart || isRunning} className="btn-secondary">
              <RotateCcw className="h-3.5 w-3.5" /> Re-scan
            </button>
          </div>
        </div>
      </div>

      <ImageGrid folderId={folderId} selectedImageId={selectedImageId} onSelectImage={(img) => onSelectImage(img.image_id)} />
    </div>
  );
}

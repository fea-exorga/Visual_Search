import React from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Play,
  Pause,
  RotateCcw,
  Zap,
  Clock,
  ImageIcon,
  FolderOpen,
} from 'lucide-react';
import { getStats, getFolders, startIndex, pauseIndex, resumeIndex } from '../hooks/useApi';
import { useSSE } from '../hooks/useSSE';
import ImageGrid from './ImageGrid';
import type { ImageSummary, IndexProgress } from '../types';

interface CenterPanelProps {
  folderId: number | null;
  selectedImageId: number | null;
  onSelectImage: (image: ImageSummary) => void;
  jobId: string | null;
  onJobStarted: (jobId: string) => void;
}

function formatEta(sec: number | null | undefined): string {
  if (sec == null || sec <= 0) return '--:--';
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

function progressStages(p: IndexProgress | null) {
  if (!p || p.total === 0) {
    return [
      { label: '1. File Scan', pct: 0 },
      { label: '2. Color Palette', pct: 0 },
      { label: '3. OCR Text', pct: 0 },
      { label: '4. Vision Embeddings', pct: 0 },
    ];
  }
  const t = p.total;
  return [
    { label: '1. File Scan', pct: Math.round((p.scanned / t) * 100) },
    { label: '2. Color Palette', pct: Math.round((p.color_done / t) * 100) },
    { label: '3. OCR Text', pct: Math.round((p.ocr_done / t) * 100) },
    { label: '4. Vision Embeddings', pct: Math.round((p.embedded / t) * 100) },
  ];
}

export default function CenterPanel({
  folderId,
  selectedImageId,
  onSelectImage,
  jobId,
  onJobStarted,
}: CenterPanelProps) {
  const queryClient = useQueryClient();

  const { data: stats } = useQuery({
    queryKey: ['stats'],
    queryFn: getStats,
    refetchInterval: 10000,
  });

  const { data: folders = [] } = useQuery({
    queryKey: ['folders'],
    queryFn: getFolders,
    refetchInterval: 5000,
  });

  const { progress, isConnected } = useSSE(jobId);
  const isRunning = jobId !== null && isConnected;

  const startMutation = useMutation({
    mutationFn: (forceReindex: boolean) => {
      const folderIds = folders.map((f) => f.folder_id);
      return startIndex(folderIds, forceReindex);
    },
    onSuccess: (data) => {
      onJobStarted(data.job_id);
      queryClient.invalidateQueries({ queryKey: ['folders'] });
    },
  });

  const pauseMutation = useMutation({
    mutationFn: () => pauseIndex(jobId!),
  });

  const resumeMutation = useMutation({
    mutationFn: () => resumeIndex(jobId!),
  });

  const stages = progressStages(progress);

  return (
    <div className="flex flex-1 flex-col overflow-y-auto">
      {/* Pipeline Status Banner */}
      <div className="sticky top-0 z-10 border-b border-slate-700/40 bg-slate-950/95 px-6 py-4 backdrop-blur-sm">
        {/* Progress bar */}
        <div className="mb-3 flex items-center gap-1">
          {stages.map((stage, i) => (
            <React.Fragment key={stage.label}>
              <div className="flex-1">
                <div className="mb-1 text-center text-[10px] font-medium text-slate-500">
                  {stage.label}
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-slate-800">
                  <div
                    className="h-full rounded-full bg-indigo-500 transition-all duration-500"
                    style={{ width: `${stage.pct}%` }}
                  />
                </div>
                <div className="mt-0.5 text-center text-[10px] text-slate-600">
                  {stage.pct}%
                </div>
              </div>
              {i < stages.length - 1 && (
                <span className="mt-3 text-xs text-slate-600">→</span>
              )}
            </React.Fragment>
          ))}
        </div>

        {/* Stats and actions row */}
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-4 text-xs text-slate-400">
            {isRunning && progress ? (
              <>
                <span className="flex items-center gap-1">
                  <Zap className="h-3.5 w-3.5 text-amber-400" />
                  {progress.images_per_sec?.toFixed(1) ?? '—'} img/s
                </span>
                <span className="flex items-center gap-1">
                  <Clock className="h-3.5 w-3.5 text-blue-400" />
                  ~{formatEta(progress.eta_sec)} remaining
                </span>
                <span className="text-slate-600">
                  {progress.failed > 0 && (
                    <span className="text-red-400">{progress.failed} failed</span>
                  )}
                </span>
              </>
            ) : (
              <>
                <span className="flex items-center gap-1">
                  <ImageIcon className="h-3.5 w-3.5" />
                  {stats?.total_images ?? 0} images indexed
                </span>
                <span className="flex items-center gap-1">
                  <FolderOpen className="h-3.5 w-3.5" />
                  across {stats?.total_folders ?? 0} folders
                </span>
              </>
            )}
          </div>

          <div className="flex items-center gap-2">
            {!isRunning ? (
              <button
                onClick={() => startMutation.mutate(false)}
                disabled={folders.length === 0 || startMutation.isPending}
                className="flex items-center gap-1.5 rounded-lg bg-emerald-600 px-4 py-1.5 text-xs font-medium text-white transition-colors hover:bg-emerald-500 disabled:opacity-40"
              >
                <Play className="h-3.5 w-3.5" />
                Start
              </button>
            ) : (
              <button
                onClick={() => pauseMutation.mutate()}
                disabled={pauseMutation.isPending}
                className="flex items-center gap-1.5 rounded-lg bg-amber-600 px-4 py-1.5 text-xs font-medium text-white transition-colors hover:bg-amber-500 disabled:opacity-40"
              >
                <Pause className="h-3.5 w-3.5" />
                Pause
              </button>
            )}
            <button
              onClick={() => startMutation.mutate(true)}
              disabled={folders.length === 0 || startMutation.isPending}
              className="flex items-center gap-1.5 rounded-lg bg-blue-600 px-4 py-1.5 text-xs font-medium text-white transition-colors hover:bg-blue-500 disabled:opacity-40"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              Re-scan
            </button>
          </div>
        </div>
      </div>

      {/* Image Grid */}
      <ImageGrid
        folderId={folderId}
        selectedImageId={selectedImageId}
        onSelectImage={onSelectImage}
      />
    </div>
  );
}

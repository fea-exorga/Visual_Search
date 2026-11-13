import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  FolderOpen,
  Plus,
  Trash2,
  ChevronDown,
  ChevronRight,
  Cpu,
  Monitor,
  Play,
  Pause,
  RotateCcw,
  Zap,
} from 'lucide-react';
import { getFolders, addFolder, deleteFolder, getSystemInfo } from '../hooks/useApi';
import type { IndexingState } from '../hooks/useIndexing';
import type { Folder } from '../types';
import { basename } from '../lib/format';

function statusBadge(folder: Folder) {
  const { total = 0, completed = 0, error = 0 } = folder.image_counts ?? ({} as Folder['image_counts']);
  if (total === 0) {
    return <span className="rounded-full bg-slate-800 px-1.5 py-0.5 text-[10px] text-slate-400">Empty</span>;
  }
  if (completed + error >= total) {
    return (
      <span className="rounded-full bg-emerald-500/10 px-1.5 py-0.5 text-[10px] text-emerald-300">
        {error ? `${error} err` : 'Synced'}
      </span>
    );
  }
  return (
    <span className="rounded-full bg-amber-500/10 px-1.5 py-0.5 text-[10px] text-amber-300">
      {Math.round((completed / total) * 100)}%
    </span>
  );
}

interface SidebarProps {
  collapsed: boolean;
  selectedFolderId: number | null;
  onSelectFolder: (id: number | null) => void;
  indexing: IndexingState;
}

export default function Sidebar({ collapsed, selectedFolderId, onSelectFolder, indexing }: SidebarProps) {
  const queryClient = useQueryClient();
  const [showAddInput, setShowAddInput] = useState(false);
  const [newPath, setNewPath] = useState('');
  const [infoOpen, setInfoOpen] = useState(false);

  const { data: allFolders = [], isError: foldersError } = useQuery({
    queryKey: ['folders'],
    queryFn: getFolders,
    refetchInterval: 5000,
  });
  const folders = allFolders.filter((f) => Boolean(f.enabled));

  const { data: info } = useQuery({ queryKey: ['systemInfo'], queryFn: getSystemInfo, staleTime: 60_000, retry: 1 });

  const addMutation = useMutation({
    mutationFn: (path: string) => addFolder(path),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['folders'] });
      setShowAddInput(false);
      setNewPath('');
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (id: number) => deleteFolder(id),
    onSuccess: (_d, id) => {
      queryClient.invalidateQueries({ queryKey: ['folders'] });
      if (selectedFolderId === id) onSelectFolder(null);
    },
  });

  const handleAdd = () => {
    const trimmed = newPath.trim();
    if (trimmed) addMutation.mutate(trimmed);
  };

  const p = indexing.progress;
  const pct = p && p.total ? Math.round((Math.max(p.embedded, p.ocr_done, p.color_done) / p.total) * 100) : 0;
  const vectors = info?.vector_models?.reduce((s, m) => s + m.count, 0) ?? 0;

  return (
    <aside
      className={`flex h-full shrink-0 flex-col border-r border-slate-800/80 bg-slate-950/70 backdrop-blur transition-[width] duration-200 ${
        collapsed ? 'w-14' : 'w-[240px]'
      }`}
    >
      {/* Watched Folders */}
      <div className={`flex-1 overflow-y-auto py-3 ${collapsed ? 'px-1.5' : 'px-3'}`}>
        {!collapsed && (
          <h2 className="mb-2 px-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">Watched folders</h2>
        )}

        <div className="space-y-0.5">
          <button
            onClick={() => onSelectFolder(null)}
            title="All folders"
            className={`flex w-full items-center gap-2 rounded-lg py-2 text-left text-sm transition-colors ${
              collapsed ? 'justify-center px-0' : 'px-2.5'
            } ${selectedFolderId === null ? 'bg-indigo-500/15 text-indigo-200' : 'text-slate-300 hover:bg-slate-800/70'}`}
          >
            <Monitor className="h-4 w-4 shrink-0" />
            {!collapsed && <span>All folders</span>}
          </button>

          {folders.map((folder) => (
            <div
              key={folder.folder_id}
              className={`group flex w-full items-center gap-1 rounded-lg text-left text-sm transition-colors ${
                selectedFolderId === folder.folder_id ? 'bg-indigo-500/15 text-indigo-200' : 'text-slate-300 hover:bg-slate-800/70'
              }`}
            >
              <button
                className={`flex min-w-0 flex-1 items-center gap-2 py-2 ${collapsed ? 'justify-center' : 'pl-2.5'}`}
                onClick={() => onSelectFolder(folder.folder_id)}
                title={folder.path}
              >
                <FolderOpen className="h-4 w-4 shrink-0 text-slate-400" />
                {!collapsed && (
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between gap-1">
                      <span className="truncate text-[13px]">{basename(folder.path)}</span>
                      {statusBadge(folder)}
                    </div>
                    <p className="text-[11px] text-slate-500">{folder.image_counts?.total ?? 0} images</p>
                  </div>
                )}
              </button>
              {!collapsed && (
                <button
                  onClick={() => deleteMutation.mutate(folder.folder_id)}
                  className="mr-1.5 rounded p-1 text-slate-500 opacity-0 transition hover:text-rose-400 focus:opacity-100 group-hover:opacity-100"
                  title="Remove folder"
                  aria-label={`Remove folder ${basename(folder.path)}`}
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              )}
            </div>
          ))}
          {foldersError && !collapsed && <p className="px-2.5 py-1 text-xs text-rose-400/80">Backend unreachable</p>}
        </div>

        {/* Add Folder */}
        {showAddInput && !collapsed ? (
          <div className="mt-2 space-y-2 px-1">
            <input
              type="text"
              value={newPath}
              onChange={(e) => setNewPath(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleAdd();
                if (e.key === 'Escape') setShowAddInput(false);
              }}
              placeholder="/path/to/images"
              autoFocus
              className="input text-xs"
            />
            <div className="flex gap-2">
              <button onClick={handleAdd} disabled={addMutation.isPending} className="btn-primary flex-1 py-1.5">
                {addMutation.isPending ? 'Adding…' : 'Add'}
              </button>
              <button
                onClick={() => {
                  setShowAddInput(false);
                  setNewPath('');
                }}
                className="btn-ghost"
              >
                Cancel
              </button>
            </div>
            {addMutation.isError && <p className="text-xs text-rose-400">{(addMutation.error as Error).message}</p>}
          </div>
        ) : (
          <button
            onClick={() => setShowAddInput(true)}
            title="Add folder"
            className={`mt-1 flex w-full items-center gap-2 rounded-lg py-2 text-sm text-slate-400 transition-colors hover:bg-slate-800/70 hover:text-slate-200 ${
              collapsed ? 'justify-center' : 'px-2.5'
            }`}
            disabled={collapsed}
          >
            <Plus className="h-4 w-4" />
            {!collapsed && <span>Add folder</span>}
          </button>
        )}
      </div>

      {/* Indexing */}
      <div className={`border-t border-slate-800/80 py-3 ${collapsed ? 'px-1.5' : 'px-3'}`}>
        {!collapsed && (
          <div className="mb-2 flex items-center justify-between px-1">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Indexing</span>
            {indexing.isRunning && p && (
              <span className="flex items-center gap-1 text-[10px] text-amber-300">
                <Zap className="h-3 w-3" />
                {p.images_per_sec?.toFixed(1) ?? '—'} img/s
              </span>
            )}
          </div>
        )}
        {indexing.isRunning && !collapsed && (
          <div className="mb-2 px-1">
            <div className="h-1.5 overflow-hidden rounded-full bg-slate-800">
              <div className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-fuchsia-500 transition-all" style={{ width: `${pct}%` }} />
            </div>
            <p className="mt-1 text-[10px] text-slate-500">
              {p ? `${p.stage} · ${pct}% of ${p.total}` : 'Starting…'}
            </p>
          </div>
        )}
        <div className={`flex gap-1.5 ${collapsed ? 'flex-col items-center' : ''}`}>
          {!indexing.isRunning ? (
            <button
              onClick={() => indexing.start(false)}
              disabled={!indexing.canStart}
              className={`btn-primary ${collapsed ? 'h-9 w-9 p-0' : 'flex-1 py-1.5'}`}
              title="Index new & changed files"
            >
              <Play className="h-3.5 w-3.5" />
              {!collapsed && 'Index'}
            </button>
          ) : (
            <button
              onClick={() => (indexing.isPaused ? indexing.resume() : indexing.pause())}
              className={`btn-secondary ${collapsed ? 'h-9 w-9 p-0' : 'flex-1'}`}
              title={indexing.isPaused ? 'Resume' : 'Pause'}
            >
              {indexing.isPaused ? <Play className="h-3.5 w-3.5" /> : <Pause className="h-3.5 w-3.5" />}
              {!collapsed && (indexing.isPaused ? 'Resume' : 'Pause')}
            </button>
          )}
          <button
            onClick={() => indexing.start(true)}
            disabled={!indexing.canStart || indexing.isRunning}
            className={`btn-secondary ${collapsed ? 'h-9 w-9 p-0' : ''}`}
            title="Force re-index everything"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            {!collapsed && 'Re-scan'}
          </button>
        </div>
        {indexing.error && !collapsed && <p className="mt-1.5 px-1 text-[10px] text-rose-400">{indexing.error}</p>}
      </div>

      {/* System info */}
      {!collapsed && (
        <div className="border-t border-slate-800/80 px-3 py-2">
          <button
            onClick={() => setInfoOpen(!infoOpen)}
            aria-expanded={infoOpen}
            className="flex w-full items-center justify-between rounded-lg px-1 py-1 text-slate-400 transition-colors hover:text-slate-200"
          >
            <span className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider">
              <Cpu className="h-3.5 w-3.5" /> Engine
            </span>
            <span className="flex items-center gap-1 text-[10px] normal-case text-slate-500">
              {info ? `${info.embed_backend}${info.embed_backend === 'onnx' ? `·${info.onnx_precision}` : ''}` : '—'}
              {infoOpen ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
            </span>
          </button>
          {infoOpen && (
            <dl className="mt-1.5 grid grid-cols-[auto,1fr] gap-x-3 gap-y-1 px-1 pb-1 text-[11px]">
              {info ? (
                <>
                  <dt className="text-slate-500">Encoder</dt>
                  <dd className="truncate text-right text-slate-300">{info.embed_backend}</dd>
                  <dt className="text-slate-500">Precision</dt>
                  <dd className="text-right text-slate-300">{info.onnx_precision}</dd>
                  <dt className="text-slate-500">Dim</dt>
                  <dd className="text-right text-slate-300">{info.embedding_dim}-d</dd>
                  <dt className="text-slate-500">OCR</dt>
                  <dd className="text-right text-slate-300">{info.ocr_engine}</dd>
                  <dt className="text-slate-500">Vectors</dt>
                  <dd className="text-right text-slate-300">{vectors.toLocaleString()}</dd>
                  {info.vector_models?.map((m) => (
                    <React.Fragment key={m.model}>
                      <dt className="col-span-2 truncate font-mono text-[10px] text-slate-600" title={m.model}>
                        {m.model} · {m.count}
                      </dt>
                    </React.Fragment>
                  ))}
                </>
              ) : (
                <dd className="col-span-2 text-slate-500">System info unavailable</dd>
              )}
            </dl>
          )}
        </div>
      )}
    </aside>
  );
}

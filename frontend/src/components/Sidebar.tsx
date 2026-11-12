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
  ScanSearch,
} from 'lucide-react';
import { getFolders, addFolder, deleteFolder } from '../hooks/useApi';
import type { Folder } from '../types';

function folderBasename(path: string): string {
  return path.split('/').filter(Boolean).pop() || path;
}

function statusBadge(folder: Folder) {
  const total = folder.total_images || 0;
  const completed = folder.completed || 0;
  if (total === 0) {
    return (
      <span className="inline-flex items-center rounded-full bg-slate-700 px-2 py-0.5 text-xs text-slate-300">
        Empty
      </span>
    );
  }
  if (completed === total) {
    return (
      <span className="inline-flex items-center rounded-full bg-emerald-900/60 px-2 py-0.5 text-xs text-emerald-400">
        Synced
      </span>
    );
  }
  const pct = Math.round((completed / total) * 100);
  return (
    <span className="inline-flex items-center rounded-full bg-amber-900/60 px-2 py-0.5 text-xs text-amber-400">
      Indexing {pct}%
    </span>
  );
}

interface SidebarProps {
  selectedFolderId: number | null;
  onSelectFolder: (id: number | null) => void;
}

export default function Sidebar({ selectedFolderId, onSelectFolder }: SidebarProps) {
  const queryClient = useQueryClient();
  const [showAddInput, setShowAddInput] = useState(false);
  const [newPath, setNewPath] = useState('');
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [visionModel, setVisionModel] = useState('SigLIP-Base-224');
  const [ocrEngine, setOcrEngine] = useState<'EasyOCR' | 'Tesseract'>('EasyOCR');

  const { data: folders = [] } = useQuery({
    queryKey: ['folders'],
    queryFn: getFolders,
    refetchInterval: 5000,
  });

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
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['folders'] });
      if (selectedFolderId !== null) onSelectFolder(null);
    },
  });

  const handleAdd = () => {
    const trimmed = newPath.trim();
    if (trimmed) addMutation.mutate(trimmed);
  };

  return (
    <aside className="flex h-full w-[260px] min-w-[260px] flex-col border-r border-slate-700/60 bg-slate-900 max-md:w-14 max-md:min-w-14">
      {/* App Title */}
      <div className="border-b border-slate-700/40 px-5 py-5 max-md:px-2 max-md:py-3">
        <div className="flex items-center gap-2">
          <ScanSearch className="h-6 w-6 text-indigo-400" />
          <h1 className="text-lg font-bold text-slate-100 max-md:hidden">PixelIndex</h1>
        </div>
        <p className="mt-0.5 text-xs text-slate-400 max-md:hidden">
          Multimodal Desktop Search
        </p>
      </div>

      {/* Watched Folders */}
      <div className="flex-1 overflow-y-auto px-3 py-4 max-md:px-1">
        <h2 className="mb-3 px-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500 max-md:hidden">
          Watched Folders
        </h2>

        <div className="space-y-1">
          {/* All folders option */}
          <button
            onClick={() => onSelectFolder(null)}
            className={`flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm transition-colors duration-200 max-md:justify-center max-md:px-2 ${
              selectedFolderId === null
                ? 'bg-indigo-600/20 text-indigo-300'
                : 'text-slate-300 hover:bg-slate-800'
            }`}
          >
            <Monitor className="h-4 w-4 shrink-0" />
            <span className="max-md:hidden">All Folders</span>
          </button>

          {folders.map((folder) => (
            <div
              key={folder.folder_id}
              className={`group flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm transition-colors duration-200 max-md:justify-center max-md:px-2 ${
                selectedFolderId === folder.folder_id
                  ? 'bg-indigo-600/20 text-indigo-300'
                  : 'text-slate-300 hover:bg-slate-800'
              }`}
            >
              <button
                className="flex min-w-0 flex-1 items-center gap-2 max-md:justify-center"
                onClick={() => onSelectFolder(folder.folder_id)}
              >
                <FolderOpen className="h-4 w-4 shrink-0 text-slate-400" />
                <div className="min-w-0 flex-1 max-md:hidden">
                  <div className="flex items-center justify-between gap-1">
                    <span className="truncate text-sm">{folderBasename(folder.path)}</span>
                    {statusBadge(folder)}
                  </div>
                  <p className="text-xs text-slate-500">
                    {folder.total_images || 0} images
                  </p>
                </div>
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  deleteMutation.mutate(folder.folder_id);
                }}
                className="hidden text-slate-500 transition-colors hover:text-red-400 group-hover:block max-md:hidden"
                title="Remove folder"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            </div>
          ))}
        </div>

        {/* Add Folder */}
        {showAddInput ? (
          <div className="mt-2 space-y-2 px-1 max-md:hidden">
            <input
              type="text"
              value={newPath}
              onChange={(e) => setNewPath(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleAdd()}
              placeholder="/path/to/images"
              autoFocus
              className="w-full rounded-md border border-slate-600 bg-slate-800 px-3 py-1.5 text-sm text-slate-100 placeholder:text-slate-500 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            />
            <div className="flex gap-2">
              <button
                onClick={handleAdd}
                disabled={addMutation.isPending}
                className="flex-1 rounded-md bg-indigo-600 px-3 py-1 text-xs font-medium text-white transition-colors hover:bg-indigo-500 disabled:opacity-50"
              >
                {addMutation.isPending ? 'Adding...' : 'Add'}
              </button>
              <button
                onClick={() => {
                  setShowAddInput(false);
                  setNewPath('');
                }}
                className="rounded-md px-3 py-1 text-xs text-slate-400 transition-colors hover:bg-slate-800"
              >
                Cancel
              </button>
            </div>
            {addMutation.isError && (
              <p className="text-xs text-red-400">
                {(addMutation.error as Error).message}
              </p>
            )}
          </div>
        ) : (
          <button
            onClick={() => setShowAddInput(true)}
            className="mt-2 flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm text-slate-400 transition-colors hover:bg-slate-800 hover:text-slate-200 max-md:justify-center max-md:px-2"
          >
            <Plus className="h-4 w-4" />
            <span className="max-md:hidden">Add Folder</span>
          </button>
        )}
      </div>

      {/* Pipeline Settings */}
      <div className="border-t border-slate-700/40 px-3 py-3 max-md:hidden">
        <button
          onClick={() => setSettingsOpen(!settingsOpen)}
          className="flex w-full items-center justify-between rounded-lg px-2 py-1.5 text-sm text-slate-400 transition-colors hover:text-slate-200"
        >
          <span className="text-[11px] font-semibold uppercase tracking-wider">
            Pipeline Settings
          </span>
          {settingsOpen ? (
            <ChevronDown className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
        </button>

        {settingsOpen && (
          <div className="mt-2 space-y-3 px-2">
            {/* Vision Model */}
            <div>
              <label className="mb-1 block text-xs text-slate-500">Vision Model</label>
              <select
                value={visionModel}
                onChange={(e) => setVisionModel(e.target.value)}
                className="w-full rounded-md border border-slate-600 bg-slate-800 px-2 py-1.5 text-xs text-slate-200 focus:border-indigo-500 focus:outline-none"
              >
                <option>SigLIP-Base-224</option>
                <option>CLIP-ViT-B/32</option>
              </select>
            </div>

            {/* OCR Engine */}
            <div>
              <label className="mb-1 block text-xs text-slate-500">OCR Engine</label>
              <div className="flex rounded-md border border-slate-600 bg-slate-800">
                {(['EasyOCR', 'Tesseract'] as const).map((engine) => (
                  <button
                    key={engine}
                    onClick={() => setOcrEngine(engine)}
                    className={`flex-1 px-2 py-1.5 text-xs transition-colors ${
                      ocrEngine === engine
                        ? 'bg-indigo-600 text-white'
                        : 'text-slate-400 hover:text-slate-200'
                    } ${engine === 'EasyOCR' ? 'rounded-l-md' : 'rounded-r-md'}`}
                  >
                    {engine}
                  </button>
                ))}
              </div>
            </div>

            {/* Device Indicator */}
            <div className="flex items-center gap-2">
              <Cpu className="h-4 w-4 text-slate-500" />
              <span className="text-xs text-slate-400">Device:</span>
              <span className="flex items-center gap-1 text-xs text-slate-200">
                <span className="inline-block h-2 w-2 rounded-full bg-emerald-500" />
                MPS (Apple Silicon)
              </span>
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}

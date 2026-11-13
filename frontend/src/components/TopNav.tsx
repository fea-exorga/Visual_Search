import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Search, Orbit, Copy, LayoutGrid, PanelLeft, ScanSearch, Zap } from 'lucide-react';
import { getStats } from '../hooks/useApi';
import type { IndexingState } from '../hooks/useIndexing';
import type { ViewName } from '../types';

export const VIEWS: { id: ViewName; label: string; icon: React.ComponentType<{ className?: string }> }[] = [
  { id: 'search', label: 'Search', icon: Search },
  { id: 'galaxy', label: 'Galaxy', icon: Orbit },
  { id: 'duplicates', label: 'Duplicates', icon: Copy },
  { id: 'library', label: 'Library', icon: LayoutGrid },
];

interface TopNavProps {
  view: ViewName;
  onChangeView: (v: ViewName) => void;
  onToggleSidebar: () => void;
  indexing: IndexingState;
}

export default function TopNav({ view, onChangeView, onToggleSidebar, indexing }: TopNavProps) {
  const stats = useQuery({ queryKey: ['stats'], queryFn: getStats, refetchInterval: 10000, retry: 0 });
  const p = indexing.progress;
  const pct = p && p.total ? Math.round((p.embedded / p.total) * 100) : 0;

  const onKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
    const i = VIEWS.findIndex((v) => v.id === view);
    const next = VIEWS[(i + (e.key === 'ArrowRight' ? 1 : VIEWS.length - 1)) % VIEWS.length];
    onChangeView(next.id);
    (e.currentTarget.querySelector(`[data-view="${next.id}"]`) as HTMLElement | null)?.focus();
  };

  return (
    <header className="flex h-14 shrink-0 items-center gap-3 border-b border-slate-800/80 bg-slate-950/70 px-3 backdrop-blur-md">
      <button onClick={onToggleSidebar} className="btn-ghost p-2" aria-label="Toggle sidebar" title="Toggle sidebar ( [ )">
        <PanelLeft className="h-4 w-4" />
      </button>
      <div className="flex items-center gap-2 pr-2">
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-500 to-fuchsia-500 shadow-lg shadow-indigo-900/50">
          <ScanSearch className="h-4 w-4 text-white" />
        </div>
        <span className="hidden text-[15px] font-semibold tracking-tight text-slate-100 lg:block">
          Pixel<span className="text-indigo-300">Index</span>
        </span>
      </div>

      <nav
        role="tablist"
        aria-label="Views"
        onKeyDown={onKeyDown}
        className="flex items-center gap-0.5 rounded-xl border border-slate-800 bg-slate-900/70 p-1"
      >
        {VIEWS.map((v, i) => {
          const active = v.id === view;
          const Icon = v.icon;
          return (
            <button
              key={v.id}
              role="tab"
              data-view={v.id}
              aria-selected={active}
              tabIndex={active ? 0 : -1}
              onClick={() => onChangeView(v.id)}
              title={`${v.label} (${i + 1})`}
              className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                active
                  ? 'bg-gradient-to-r from-indigo-500/90 to-violet-500/90 text-white shadow shadow-indigo-900/50'
                  : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
              }`}
            >
              <Icon className="h-3.5 w-3.5" />
              {v.label}
              <kbd className={`ml-0.5 hidden font-mono text-[9px] xl:inline ${active ? 'text-indigo-200' : 'text-slate-600'}`}>{i + 1}</kbd>
            </button>
          );
        })}
      </nav>

      <div className="ml-auto flex items-center gap-3 text-xs">
        {indexing.isRunning && (
          <span className="flex items-center gap-2 rounded-full border border-amber-500/30 bg-amber-500/10 px-2.5 py-1 text-amber-200">
            <Zap className="h-3 w-3" />
            Indexing {pct}%
            <span className="h-1 w-16 overflow-hidden rounded-full bg-amber-900/60">
              <span className="block h-full bg-amber-400 transition-all" style={{ width: `${pct}%` }} />
            </span>
          </span>
        )}
        <span
          className="flex items-center gap-1.5 text-slate-400"
          title={stats.isError ? 'Backend unreachable' : 'Backend connected'}
        >
          <span
            className={`h-2 w-2 rounded-full ${
              stats.isError ? 'bg-rose-500' : stats.data ? 'bg-emerald-400 shadow-[0_0_8px] shadow-emerald-400/70' : 'bg-slate-600'
            }`}
          />
          {stats.isError ? 'offline' : stats.data ? `${stats.data.total_images.toLocaleString()} images` : '…'}
        </span>
      </div>
    </header>
  );
}

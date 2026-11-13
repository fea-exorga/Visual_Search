import React, { useMemo, useState } from 'react';
import { keepPreviousData, useQuery } from '@tanstack/react-query';
import { Copy, HardDrive, Layers, CheckCircle2, Crown } from 'lucide-react';
import { getDuplicates, thumbnailUrl } from '../../hooks/useApi';
import { useDebounce } from '../../hooks/useDebounce';
import { basename, dirname, formatBytes } from '../../lib/format';
import type { DuplicateGroup } from '../../types';
import { CenteredSpinner, EmptyState, ErrorState, Spinner } from '../ui/States';
import Slider from '../ui/Slider';

interface DuplicatesViewProps {
  selectedImageId: number | null;
  onSelectImage: (id: number) => void;
}

type KindFilter = 'all' | 'exact' | 'near';

const PREVIEW = 12;

/** Keep the largest file of a group; everything else is reclaimable. */
function groupStats(g: DuplicateGroup) {
  let keep = g.images[0];
  for (const im of g.images) if ((im.file_size ?? 0) > (keep?.file_size ?? 0)) keep = im;
  const total = g.images.reduce((s, im) => s + (im.file_size ?? 0), 0);
  return { keepId: keep?.image_id, reclaim: total - (keep?.file_size ?? 0) };
}

export default function DuplicatesView({ selectedImageId, onSelectImage }: DuplicatesViewProps) {
  const [threshold, setThreshold] = useState(0.95);
  const [maxHamming, setMaxHamming] = useState(6);
  const [kind, setKind] = useState<KindFilter>('all');
  const [expanded, setExpanded] = useState<Set<number>>(new Set());
  const dThreshold = useDebounce(threshold, 300);
  const dHamming = useDebounce(maxHamming, 300);

  const q = useQuery({
    queryKey: ['duplicates', dThreshold, dHamming],
    queryFn: ({ signal }) => getDuplicates(dThreshold, dHamming, signal),
    placeholderData: keepPreviousData,
    retry: false,
  });

  const groups = useMemo(() => {
    const gs = (q.data?.groups ?? []).map((g) => ({ g, ...groupStats(g) }));
    return gs.sort((a, b) => b.reclaim - a.reclaim);
  }, [q.data]);
  const shown = groups.filter(({ g }) => kind === 'all' || g.kind === kind);
  const totalReclaim = groups.reduce((s, x) => s + x.reclaim, 0);
  const nExact = groups.filter((x) => x.g.kind === 'exact').length;

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-y-auto">
      <div className="sticky top-0 z-10 border-b border-slate-800/60 bg-slate-950/85 px-5 py-4 backdrop-blur-md">
        <div className="flex flex-wrap items-end gap-x-8 gap-y-3">
          <div className="w-56">
            <Slider
              label="Cosine threshold"
              value={threshold}
              min={0.85}
              max={0.99}
              step={0.01}
              onChange={setThreshold}
              format={(v) => v.toFixed(2)}
            />
          </div>
          <div className="w-48">
            <Slider label="Max pHash Hamming" value={maxHamming} min={0} max={16} step={1} onChange={setMaxHamming} />
          </div>
          <div className="seg" role="group" aria-label="Duplicate kind">
            {(['all', 'exact', 'near'] as const).map((k) => (
              <button key={k} aria-pressed={kind === k} onClick={() => setKind(k)} className="capitalize">
                {k}
              </button>
            ))}
          </div>
          {q.isFetching && <Spinner className="mb-1 h-4 w-4" />}

          {q.data && (
            <div className="ml-auto flex gap-3">
              <Stat icon={<Layers className="h-4 w-4" />} label="Groups" value={`${q.data.n_groups}`} sub={`${nExact} exact`} />
              <Stat icon={<Copy className="h-4 w-4" />} label="Images" value={`${q.data.n_images}`} sub="in groups" />
              <Stat
                icon={<HardDrive className="h-4 w-4" />}
                label="Reclaimable"
                value={formatBytes(totalReclaim)}
                sub="keeping largest"
                accent
              />
            </div>
          )}
        </div>
      </div>

      <div className="flex flex-1 flex-col px-5 py-4">
        {q.isLoading ? (
          <CenteredSpinner label="Scanning for duplicates…" />
        ) : q.isError && !q.data ? (
          <ErrorState error={q.error} onRetry={() => q.refetch()} />
        ) : shown.length === 0 ? (
          <EmptyState icon={<CheckCircle2 className="h-6 w-6" />} title="No duplicates found">
            Nothing above a cosine similarity of {threshold.toFixed(2)} or within Hamming distance {maxHamming}. Lower the
            threshold to find looser matches.
          </EmptyState>
        ) : (
          <div className={`grid grid-cols-1 gap-3 2xl:grid-cols-2 ${q.isPlaceholderData ? 'opacity-60' : ''}`}>
            {shown.map(({ g, keepId, reclaim }) => (
              <article key={g.group_id} className="panel min-w-0 p-3">
                <header className="mb-2.5 flex items-center gap-2 text-xs">
                  <span
                    className={`chip py-0.5 ${
                      g.kind === 'exact'
                        ? 'border-rose-400/30 bg-rose-500/10 text-rose-200'
                        : 'border-sky-400/30 bg-sky-500/10 text-sky-200'
                    }`}
                  >
                    {g.kind === 'exact' ? 'Exact' : 'Near'}
                  </span>
                  <span className="text-slate-400">
                    {g.images.length} images · group #{g.group_id}
                  </span>
                  <span className="ml-auto font-mono text-emerald-300">−{formatBytes(reclaim)}</span>
                </header>
                <div className="flex gap-2.5 overflow-x-auto pb-1">
                  {(expanded.has(g.group_id) ? g.images : g.images.slice(0, PREVIEW)).map((im) => {
                    const keep = im.image_id === keepId;
                    return (
                      <button
                        key={im.image_id}
                        onClick={() => onSelectImage(im.image_id)}
                        title={im.file_path}
                        className={`group w-36 shrink-0 overflow-hidden rounded-lg border bg-slate-950/60 text-left transition hover:border-slate-500 ${
                          selectedImageId === im.image_id ? 'border-indigo-400 ring-2 ring-indigo-500/30' : 'border-slate-800'
                        }`}
                      >
                        <div className="relative aspect-square bg-slate-900">
                          <img
                            src={im.thumbnail_url || thumbnailUrl(im.image_id)}
                            alt={basename(im.file_path)}
                            loading="lazy"
                            className="h-full w-full object-cover"
                          />
                          {keep && (
                            <span className="absolute left-1.5 top-1.5 flex items-center gap-1 rounded-md bg-emerald-500/90 px-1.5 py-0.5 text-[9px] font-bold uppercase text-emerald-950">
                              <Crown className="h-2.5 w-2.5" /> keep
                            </span>
                          )}
                          <span className="absolute bottom-1.5 right-1.5 rounded-md bg-slate-950/80 px-1.5 py-0.5 font-mono text-[10px] text-slate-200">
                            {(im.similarity * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="space-y-0.5 px-2 py-1.5">
                          <p className="truncate text-[11px] text-slate-200">{basename(im.file_path)}</p>
                          <p className="truncate text-[10px] text-slate-500">{dirname(im.file_path)}</p>
                          <p className="flex justify-between text-[10px] text-slate-400">
                            <span>{formatBytes(im.file_size)}</span>
                            <span>
                              {im.width}×{im.height}
                            </span>
                          </p>
                        </div>
                      </button>
                    );
                  })}
                  {g.images.length > PREVIEW && !expanded.has(g.group_id) && (
                    <button
                      onClick={() => setExpanded((s) => new Set(s).add(g.group_id))}
                      className="flex w-24 shrink-0 flex-col items-center justify-center gap-1 rounded-lg border border-dashed border-slate-700 text-xs text-slate-400 transition hover:border-indigo-400/60 hover:text-indigo-200"
                    >
                      <span className="text-lg font-semibold">+{g.images.length - PREVIEW}</span>
                      show all
                    </button>
                  )}
                </div>
              </article>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function Stat({ icon, label, value, sub, accent }: { icon: React.ReactNode; label: string; value: string; sub?: string; accent?: boolean }) {
  return (
    <div
      className={`flex items-center gap-2.5 rounded-xl border px-3 py-1.5 ${
        accent ? 'border-emerald-500/30 bg-emerald-500/5' : 'border-slate-800 bg-slate-900/60'
      }`}
    >
      <span className={accent ? 'text-emerald-300' : 'text-slate-500'}>{icon}</span>
      <div>
        <div className="text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
        <div className={`text-sm font-semibold ${accent ? 'text-emerald-200' : 'text-slate-100'}`}>
          {value} {sub && <span className="text-[10px] font-normal text-slate-500">{sub}</span>}
        </div>
      </div>
    </div>
  );
}

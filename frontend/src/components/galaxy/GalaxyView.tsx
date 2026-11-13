import React, { Suspense, lazy, useCallback, useEffect, useMemo, useState } from 'react';
import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Orbit, RefreshCw, Search, X, Rotate3d, AlertTriangle, Layers, ChevronRight, Sparkles } from 'lucide-react';
import { ApiError, computeGalaxy, getGalaxy, search, thumbnailUrl } from '../../hooks/useApi';
import { useDebounce } from '../../hooks/useDebounce';
import { clusterColor, formatDate } from '../../lib/format';
import type { GalaxyResponse, ProjectionDims, ProjectionMethod } from '../../types';
import { CenteredSpinner, EmptyState, ErrorState, Spinner } from '../ui/States';
import Galaxy2D from './Galaxy2D';
import type { FlyTarget, HoverInfo } from './galaxyTypes';

const Galaxy3D = lazy(() => import('./Galaxy3D'));

interface GalaxyViewProps {
  method: ProjectionMethod;
  dims: ProjectionDims;
  onChangeProjection: (method: ProjectionMethod, dims: ProjectionDims) => void;
  focus: { imageId: number; nonce: number } | null;
  onSelectImage: (id: number) => void;
}

function ProjectionPicker({
  method,
  dims,
  onChange,
}: {
  method: ProjectionMethod;
  dims: ProjectionDims;
  onChange: (m: ProjectionMethod, d: ProjectionDims) => void;
}) {
  return (
    <div className="flex items-center gap-2">
      <div className="seg" role="group" aria-label="Projection method">
        {(['umap', 'tsne'] as const).map((m) => (
          <button key={m} aria-pressed={method === m} onClick={() => onChange(m, dims)}>
            {m === 'umap' ? 'UMAP' : 't-SNE'}
          </button>
        ))}
      </div>
      <div className="seg" role="group" aria-label="Dimensions">
        {([2, 3] as const).map((d) => (
          <button key={d} aria-pressed={dims === d} onClick={() => onChange(method, d)}>
            {d}D
          </button>
        ))}
      </div>
    </div>
  );
}

let flyNonce = 1;

/** Mean + RMS radius of a set of points. */
function centroidOf(pts: { x: number; y: number; z?: number | null }[], weights?: number[]) {
  let sw = 0, x = 0, y = 0, z = 0;
  pts.forEach((p, i) => {
    const w = weights?.[i] ?? 1;
    sw += w;
    x += p.x * w;
    y += p.y * w;
    z += (p.z ?? 0) * w;
  });
  x /= sw || 1;
  y /= sw || 1;
  z /= sw || 1;
  let r2 = 0;
  pts.forEach((p) => (r2 += (p.x - x) ** 2 + (p.y - y) ** 2 + ((p.z ?? 0) - z) ** 2));
  return { x, y, z, spread: Math.max(0.05, Math.sqrt(r2 / Math.max(1, pts.length)) * 1.4) };
}

export default function GalaxyView({ method, dims, onChangeProjection, focus, onSelectImage }: GalaxyViewProps) {
  const qc = useQueryClient();
  const rootRef = React.useRef<HTMLDivElement>(null);
  const [activeClusters, setActiveClusters] = useState<Set<number>>(new Set());
  const [hover, setHover] = useState<HoverInfo | null>(null);
  const [fly, setFly] = useState<FlyTarget | null>(null);
  const [focusId, setFocusId] = useState<number | null>(null);
  const [autoRotate, setAutoRotate] = useState(true);
  const [legendOpen, setLegendOpen] = useState(true);
  const [q, setQ] = useState('');
  const [notice, setNotice] = useState<string | null>(null);
  const dq = useDebounce(q.trim(), 400);

  const galaxy = useQuery({
    queryKey: ['galaxy', method, dims],
    queryFn: ({ signal }) => getGalaxy(method, dims, signal),
    retry: (n, err) => !(err instanceof ApiError && err.status < 500) && n < 1,
    staleTime: 60_000,
  });

  const compute = useMutation({
    mutationFn: () => computeGalaxy({ method, dims }),
    onSuccess: (d) => {
      qc.setQueryData(['galaxy', method, dims], d);
      qc.invalidateQueries({ queryKey: ['galaxy'] });
    },
  });

  const data: GalaxyResponse | undefined = galaxy.data;

  const clusterColors = useMemo(() => {
    const m = new Map<number, string>();
    data?.clusters.forEach((c) => m.set(c.cluster_id, clusterColor(c.color, c.cluster_id)));
    data?.points.forEach((p) => {
      if (!m.has(p.cluster_id)) m.set(p.cluster_id, clusterColor(null, p.cluster_id));
    });
    return m;
  }, [data]);

  const clusterById = useMemo(() => new Map((data?.clusters ?? []).map((c) => [c.cluster_id, c])), [data]);
  const pointById = useMemo(() => new Map((data?.points ?? []).map((p) => [p.image_id, p])), [data]);

  // reset local selections when the dataset changes
  useEffect(() => {
    setActiveClusters(new Set());
    setHover(null);
    setFly(null);
  }, [data]);

  // ---- search integration ----
  const searchQ = useQuery({
    queryKey: ['galaxy-search', dq, method, dims],
    queryFn: ({ signal }) => search({ text: dq, k: 80, projection: { method, dims } }, signal),
    enabled: Boolean(dq) && Boolean(data),
    retry: false,
    placeholderData: keepPreviousData,
  });
  const hitScores = useMemo(() => {
    if (!dq || !searchQ.data) return null;
    const m = new Map<number, number>();
    searchQ.data.results.forEach((h) => m.set(h.image_id, h.score));
    return m;
  }, [dq, searchQ.data]);
  const highlight = useMemo(() => (hitScores ? new Set(hitScores.keys()) : null), [hitScores]);
  const hitsInGalaxy = useMemo(
    () => (hitScores ? [...hitScores.keys()].filter((id) => pointById.has(id)) : []),
    [hitScores, pointById],
  );

  useEffect(() => {
    if (!hitScores || !hitsInGalaxy.length) return;
    const pts = hitsInGalaxy.map((id) => pointById.get(id)!);
    // weight towards the best hits
    const c = centroidOf(pts.slice(0, 30), pts.slice(0, 30).map((p) => Math.max(0.05, hitScores.get(p.image_id) ?? 0)));
    setFly({ ...c, nonce: flyNonce++ });
  }, [hitScores, hitsInGalaxy, pointById]);

  // ---- inspector "Show in galaxy" ----
  useEffect(() => {
    if (!focus || !data) return;
    const p = pointById.get(focus.imageId);
    if (!p) {
      setNotice(`Image #${focus.imageId} is not part of this projection — recompute to include it.`);
      return;
    }
    setNotice(null);
    setFocusId(p.image_id);
    setActiveClusters(new Set());
    setFly({ x: p.x, y: p.y, z: p.z ?? 0, spread: 0.06, nonce: flyNonce++ });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focus?.nonce, data]);

  const toggleCluster = useCallback(
    (cid: number) => {
      setActiveClusters((prev) => {
        const next = new Set(prev);
        if (next.has(cid)) next.delete(cid);
        else {
          next.add(cid);
          const pts = data?.points.filter((p) => p.cluster_id === cid) ?? [];
          if (pts.length) setFly({ ...centroidOf(pts), nonce: flyNonce++ });
        }
        return next;
      });
    },
    [data],
  );

  const select = useCallback(
    (id: number) => {
      setFocusId(id);
      onSelectImage(id);
    },
    [onSelectImage],
  );

  // ---------- states ----------
  const notComputed = galaxy.error instanceof ApiError && galaxy.error.status === 404;

  const header = (
    <div className="pointer-events-none absolute inset-x-0 top-0 z-10 flex flex-wrap items-start gap-2 p-3">
      <div className="pointer-events-auto flex items-center gap-2 rounded-xl border border-slate-700/60 bg-slate-950/70 p-1.5 backdrop-blur-md">
        <ProjectionPicker method={method} dims={dims} onChange={onChangeProjection} />
        {data && (
          <>
            <button
              onClick={() => compute.mutate()}
              disabled={compute.isPending}
              className={`btn-secondary ${data.stale ? '!border-amber-500/50 !bg-amber-500/10 text-amber-200' : ''}`}
              title="Recompute projection & clusters"
            >
              {compute.isPending ? <Spinner className="h-3.5 w-3.5" /> : <RefreshCw className="h-3.5 w-3.5" />}
              {data.stale ? 'Stale · Recompute' : 'Recompute'}
            </button>
            {dims === 3 && (
              <button
                onClick={() => setAutoRotate((v) => !v)}
                aria-pressed={autoRotate}
                className={`btn-secondary ${autoRotate ? '!border-indigo-500/50 !bg-indigo-500/15 text-indigo-200' : ''}`}
                title="Auto-rotate"
              >
                <Rotate3d className="h-3.5 w-3.5" />
                <span className="hidden xl:inline">Auto-rotate</span>
              </button>
            )}
          </>
        )}
      </div>

      {data && (
        <div className="pointer-events-auto flex min-w-[240px] max-w-md flex-1 items-center gap-2 rounded-xl border border-slate-700/60 bg-slate-950/70 px-3 py-1.5 backdrop-blur-md focus-within:border-indigo-500/60">
          <Search className="h-4 w-4 shrink-0 text-slate-500" />
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => e.key === 'Escape' && setQ('')}
            placeholder="Light up the galaxy… “diagrams”, “beach”"
            aria-label="Search and highlight in galaxy"
            className="min-w-0 flex-1 bg-transparent py-1 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none"
          />
          {searchQ.isFetching && <Spinner className="h-3.5 w-3.5" />}
          {dq && searchQ.data && (
            <span className="whitespace-nowrap rounded-full bg-indigo-500/20 px-2 py-0.5 text-[10px] font-medium text-indigo-200">
              {hitsInGalaxy.length} lit
            </span>
          )}
          {q && (
            <button onClick={() => setQ('')} className="text-slate-400 hover:text-slate-200" aria-label="Clear galaxy search">
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      )}
    </div>
  );

  let body: React.ReactNode;
  if (galaxy.isLoading) body = <CenteredSpinner label="Loading galaxy…" />;
  else if (notComputed || (!data && compute.isPending)) {
    body = (
      <EmptyState
        icon={<Orbit className="h-6 w-6" />}
        title="No galaxy computed yet"
        action={
          <div className="flex flex-col items-center gap-3">
            <ProjectionPicker method={method} dims={dims} onChange={onChangeProjection} />
            <button onClick={() => compute.mutate()} disabled={compute.isPending} className="btn-primary">
              {compute.isPending ? <Spinner className="h-3.5 w-3.5 border-white/30 border-t-white" /> : <Sparkles className="h-3.5 w-3.5" />}
              {compute.isPending ? 'Projecting embeddings…' : 'Compute galaxy'}
            </button>
            {compute.isError && <ErrorState error={compute.error} compact />}
          </div>
        }
      >
        Project every image embedding with {method === 'umap' ? 'UMAP' : 't-SNE'} into {dims}D and cluster them into themes.
      </EmptyState>
    );
  } else if (galaxy.isError && !data) {
    body = <ErrorState error={galaxy.error} onRetry={() => galaxy.refetch()} />;
  } else if (data && data.points.length === 0) {
    body = (
      <EmptyState icon={<Orbit className="h-6 w-6" />} title="The galaxy is empty" action={
        <button onClick={() => compute.mutate()} className="btn-secondary"><RefreshCw className="h-3.5 w-3.5" /> Recompute</button>
      }>
        No embedded images in this projection yet. Index a folder, then recompute.
      </EmptyState>
    );
  } else if (data) {
    const rendererProps = {
      data,
      clusterColors,
      highlight,
      activeClusters,
      focusId,
      fly,
      onHover: setHover,
      onSelect: select,
      autoRotate,
    };
    body = data.dims === 3 || dims === 3 ? (
      <Suspense fallback={<CenteredSpinner label="Loading 3D engine…" />}>
        <Galaxy3D key={`${data.method}-${data.dims}-${data.computed_at}`} {...rendererProps} />
      </Suspense>
    ) : (
      <Galaxy2D {...rendererProps} />
    );
  }

  const hoverCluster = hover ? clusterById.get(hover.clusterId) : undefined;
  const hoverScore = hover && hitScores ? hitScores.get(hover.imageId) : undefined;
  const sortedClusters = useMemo(() => [...(data?.clusters ?? [])].sort((a, b) => b.size - a.size), [data]);

  return (
    <div ref={rootRef} className="relative min-h-0 flex-1 overflow-hidden bg-[#020617]">
      {data && header}
      <div className="absolute inset-0 flex">{body}</div>

      {/* tooltip */}
      {hover && data && (
        <div
          className="pointer-events-none absolute z-20 w-44 overflow-hidden rounded-xl border border-slate-600/60 bg-slate-950/90 shadow-2xl shadow-black/60 backdrop-blur"
          style={{
            left: hover.sx + 196 > (rootRef.current?.clientWidth ?? 2000) - 250 ? hover.sx - 192 : hover.sx + 16,
            top: Math.max(8, Math.min(hover.sy - 60, (rootRef.current?.clientHeight ?? 2000) - 250)),
          }}
        >
          <img src={thumbnailUrl(hover.imageId)} alt="" className="aspect-square w-full bg-slate-900 object-cover" />
          <div className="space-y-0.5 px-2.5 py-2 text-[11px]">
            <div className="flex items-center gap-1.5 text-slate-200">
              <span className="h-2 w-2 shrink-0 rounded-full" style={{ backgroundColor: clusterColors.get(hover.clusterId) }} />
              <span className="truncate">{hoverCluster?.label || `Cluster ${hover.clusterId}`}</span>
            </div>
            <div className="flex justify-between text-slate-500">
              <span>#{hover.imageId}</span>
              {hoverScore != null && <span className="text-indigo-300">score {(hoverScore * 100).toFixed(0)}</span>}
              <span>click to inspect</span>
            </div>
          </div>
        </div>
      )}

      {/* legend */}
      {data && data.points.length > 0 && (
        <div className="absolute bottom-3 right-3 top-16 z-10 flex w-60 flex-col">
          <div className="flex max-h-full flex-col overflow-hidden rounded-xl border border-slate-700/60 bg-slate-950/75 backdrop-blur-md">
            <button
              onClick={() => setLegendOpen((v) => !v)}
              aria-expanded={legendOpen}
              className="flex items-center gap-2 px-3 py-2 text-left text-xs font-semibold text-slate-200"
            >
              <Layers className="h-3.5 w-3.5 text-indigo-300" />
              <span className="flex-1">
                {data.clusters.length} themes · {data.points.length.toLocaleString()} images
              </span>
              <ChevronRight className={`h-3.5 w-3.5 text-slate-500 transition ${legendOpen ? 'rotate-90' : ''}`} />
            </button>
            {legendOpen && (
              <>
                <div className="min-h-0 flex-1 overflow-y-auto border-t border-slate-800 py-1">
                  {sortedClusters.map((c) => {
                    const on = activeClusters.has(c.cluster_id);
                    const dim = activeClusters.size > 0 && !on;
                    const col = clusterColors.get(c.cluster_id);
                    return (
                      <button
                        key={c.cluster_id}
                        onClick={() => toggleCluster(c.cluster_id)}
                        aria-pressed={on}
                        className={`flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs transition hover:bg-slate-800/60 ${
                          on ? 'bg-slate-800/80' : ''
                        } ${dim ? 'opacity-40' : ''}`}
                      >
                        <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ backgroundColor: col, boxShadow: `0 0 8px ${col}` }} />
                        <span className="min-w-0 flex-1 truncate text-slate-200" title={c.label}>
                          {c.label || `Cluster ${c.cluster_id}`}
                        </span>
                        <span className="font-mono text-[10px] text-slate-500">{c.size}</span>
                      </button>
                    );
                  })}
                </div>
                <div className="flex items-center justify-between border-t border-slate-800 px-3 py-1.5 text-[10px] text-slate-500">
                  <span title={String(data.computed_at ?? '')}>{formatDate(data.computed_at)}</span>
                  {activeClusters.size > 0 && (
                    <button onClick={() => setActiveClusters(new Set())} className="text-indigo-300 hover:underline">
                      show all
                    </button>
                  )}
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* notices */}
      {(notice || (data?.stale && !compute.isPending) || (searchQ.isError && dq) || (compute.isError && data)) && (
        <div className="absolute bottom-4 left-1/2 z-20 flex -translate-x-1/2 flex-col items-center gap-2">
          {data?.stale && (
            <div className="flex items-center gap-2 rounded-full border border-amber-500/30 bg-amber-950/80 px-3 py-1.5 text-xs text-amber-200 backdrop-blur">
              <AlertTriangle className="h-3.5 w-3.5" /> New images were indexed since this galaxy was computed.
              <button onClick={() => compute.mutate()} className="font-semibold underline-offset-2 hover:underline">
                Recompute
              </button>
            </div>
          )}
          {notice && (
            <div className="flex items-center gap-2 rounded-full border border-slate-600/50 bg-slate-900/90 px-3 py-1.5 text-xs text-slate-300 backdrop-blur">
              {notice}
              <button onClick={() => setNotice(null)} aria-label="Dismiss">
                <X className="h-3 w-3" />
              </button>
            </div>
          )}
          {searchQ.isError && dq && <ErrorState error={searchQ.error} compact />}
          {compute.isError && data && <ErrorState error={compute.error} compact />}
        </div>
      )}
    </div>
  );
}

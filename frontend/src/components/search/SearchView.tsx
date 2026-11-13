import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { keepPreviousData, useQuery } from '@tanstack/react-query';
import {
  Search,
  ImagePlus,
  X,
  Type,
  Palette,
  PenTool,
  Image as ImageIcon,
  Layers,
  FolderOpen,
  Timer,
  ChevronDown,
  SlidersHorizontal,
  Sparkles,
  RotateCcw,
  Upload,
  Copy,
} from 'lucide-react';
import { getFolders, getGalaxy, search, searchSimilar, searchUpload, thumbnailUrl } from '../../hooks/useApi';
import { useDebounce } from '../../hooks/useDebounce';
import { basename, clusterColor, formatMs } from '../../lib/format';
import type { ProjectionDims, ProjectionMethod, SearchResponse } from '../../types';
import { CenteredSpinner, EmptyState, ErrorState, Spinner } from '../ui/States';
import Slider from '../ui/Slider';
import ColorPicker from './ColorPicker';
import ResultCard, { SCORE_COLORS } from './ResultCard';
import SketchPad from './SketchPad';
import {
  DEFAULT_FILTERS,
  DEFAULT_WEIGHTS,
  buildSearchRequest,
  effectiveColors,
  hasQuery,
  isPureSimilar,
  type SearchFilters,
  type UploadQuery,
} from './searchState';

interface SearchViewProps {
  filters: SearchFilters;
  setFilters: React.Dispatch<React.SetStateAction<SearchFilters>>;
  upload: UploadQuery | null;
  setUpload: (u: UploadQuery | null) => void;
  projection: { method: ProjectionMethod; dims: ProjectionDims };
  selectedImageId: number | null;
  onSelectImage: (id: number) => void;
  inputRef: React.RefObject<HTMLInputElement>;
}

const EXAMPLES = ['cozy rainy evening', 'sunset over the ocean', 'lecture slide with a diagram', 'red', 'meme with caption'];

let uploadSeq = 1;

function Section({
  title,
  icon,
  open,
  onToggle,
  badge,
  children,
}: {
  title: string;
  icon: React.ReactNode;
  open: boolean;
  onToggle: () => void;
  badge?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="panel shrink-0 overflow-hidden">
      <button
        onClick={onToggle}
        aria-expanded={open}
        className="flex w-full items-center gap-2 px-3.5 py-2.5 text-left text-xs font-semibold text-slate-200 transition hover:bg-slate-800/40"
      >
        <span className="text-indigo-300">{icon}</span>
        <span className="flex-1">{title}</span>
        {badge}
        <ChevronDown className={`h-3.5 w-3.5 text-slate-500 transition ${open ? '' : '-rotate-90'}`} />
      </button>
      {open && <div className="border-t border-slate-800/80 px-3.5 py-3">{children}</div>}
    </section>
  );
}

export default function SearchView({
  filters,
  setFilters,
  upload,
  setUpload,
  projection,
  selectedImageId,
  onSelectImage,
  inputRef,
}: SearchViewProps) {
  const [sketchOpen, setSketchOpen] = useState(true);
  const [filtersOpen, setFiltersOpen] = useState(true);
  const [panelOpen, setPanelOpen] = useState(true);
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const dragDepth = useRef(0);

  const set = useCallback(
    <K extends keyof SearchFilters>(key: K, value: SearchFilters[K]) => setFilters((f) => ({ ...f, [key]: value })),
    [setFilters],
  );

  const debouncedText = useDebounce(filters.text, 300);
  const debouncedOcr = useDebounce(filters.ocr, 300);
  const effFilters = useMemo(() => ({ ...filters, ocr: debouncedOcr }), [filters, debouncedOcr]);
  const active = hasQuery(effFilters, debouncedText);
  const pureSimilar = isPureSimilar(effFilters, debouncedText);
  const req = useMemo(() => buildSearchRequest(effFilters, debouncedText, projection), [effFilters, debouncedText, projection]);

  const searchQuery = useQuery<SearchResponse>({
    queryKey: upload
      ? ['search', 'upload', upload.id, filters.k]
      : pureSimilar
        ? ['search', 'similar', filters.imageId, filters.k]
        : ['search', req],
    queryFn: ({ signal }) =>
      upload
        ? searchUpload(upload.file, filters.k, signal)
        : pureSimilar
          ? searchSimilar(filters.imageId!, filters.k, signal)
          : search(req, signal),
    enabled: Boolean(upload) || active,
    placeholderData: keepPreviousData,
    retry: false,
    staleTime: 30_000,
  });

  const galaxyQuery = useQuery({
    queryKey: ['galaxy', projection.method, projection.dims],
    queryFn: ({ signal }) => getGalaxy(projection.method, projection.dims, signal),
    retry: false,
    staleTime: 60_000,
  });
  const clusters = galaxyQuery.data?.clusters ?? [];
  const clusterMap = useMemo(() => {
    const m = new Map<number, { label: string; color: string }>();
    clusters.forEach((c) => m.set(c.cluster_id, { label: c.label, color: clusterColor(c.color, c.cluster_id) }));
    return m;
  }, [clusters]);

  const foldersQuery = useQuery({ queryKey: ['folders'], queryFn: getFolders, refetchInterval: 10_000 });
  const folders = (foldersQuery.data ?? []).filter((f) => Boolean(f.enabled));

  // ---- upload / drop / paste ----
  const acceptFile = useCallback(
    (file: File | null | undefined) => {
      if (!file || !file.type.startsWith('image/')) return;
      if (upload) URL.revokeObjectURL(upload.previewUrl);
      setUpload({ id: uploadSeq++, file, previewUrl: URL.createObjectURL(file) });
    },
    [upload, setUpload],
  );

  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const item = [...(e.clipboardData?.items ?? [])].find((i) => i.type.startsWith('image/'));
      if (item) {
        e.preventDefault();
        acceptFile(item.getAsFile());
      }
    };
    window.addEventListener('paste', onPaste);
    return () => window.removeEventListener('paste', onPaste);
  }, [acceptFile]);

  const clearUpload = () => {
    if (upload) URL.revokeObjectURL(upload.previewUrl);
    setUpload(null);
  };

  const resetAll = () => {
    clearUpload();
    setFilters((f) => ({ ...DEFAULT_FILTERS, k: f.k, includeSketchColors: f.includeSketchColors }));
  };

  // ---- active-mode chips ----
  type Chip = { key: string; icon: React.ReactNode; label: React.ReactNode; tone: string; onRemove: () => void };
  const chips: Chip[] = [];
  if (upload)
    chips.push({ key: 'upload', icon: <Upload className="h-3 w-3" />, label: `Image · ${basename(upload.file.name || 'pasted')}`, tone: 'violet', onRemove: clearUpload });
  if (filters.text.trim())
    chips.push({ key: 'text', icon: <Sparkles className="h-3 w-3" />, label: `Vibe · “${filters.text.trim()}”`, tone: 'indigo', onRemove: () => set('text', '') });
  if (filters.imageId != null)
    chips.push({
      key: 'image',
      icon: <Copy className="h-3 w-3" />,
      label: (
        <span className="flex items-center gap-1">
          <img src={thumbnailUrl(filters.imageId)} alt="" className="h-4 w-4 rounded object-cover" />
          Similar to #{filters.imageId}
        </span>
      ),
      tone: 'violet',
      onRemove: () => set('imageId', null),
    });
  if (filters.sketchPng)
    chips.push({ key: 'sketch', icon: <PenTool className="h-3 w-3" />, label: 'Sketch', tone: 'fuchsia', onRemove: () => setFilters((f) => ({ ...f, sketchPng: null, sketchColors: [] })) });
  const effColors = effectiveColors(filters);
  if (effColors.length)
    chips.push({
      key: 'colors',
      icon: <Palette className="h-3 w-3" />,
      label: (
        <span className="flex items-center gap-1">
          Colour
          {effColors.map((c) => (
            <span key={c} className="h-2.5 w-2.5 rounded-sm ring-1 ring-white/20" style={{ backgroundColor: c }} />
          ))}
        </span>
      ),
      tone: 'pink',
      onRemove: () => setFilters((f) => ({ ...f, colors: [], includeSketchColors: false })),
    });
  if (filters.ocr.trim())
    chips.push({ key: 'ocr', icon: <Type className="h-3 w-3" />, label: `Text · “${filters.ocr.trim()}”`, tone: 'amber', onRemove: () => set('ocr', '') });
  if (filters.clusterIds.length)
    chips.push({
      key: 'clusters',
      icon: <Layers className="h-3 w-3" />,
      label: filters.clusterIds.length === 1 ? clusterMap.get(filters.clusterIds[0])?.label ?? `Cluster ${filters.clusterIds[0]}` : `${filters.clusterIds.length} clusters`,
      tone: 'emerald',
      onRemove: () => set('clusterIds', []),
    });
  if (filters.folderId != null)
    chips.push({
      key: 'folder',
      icon: <FolderOpen className="h-3 w-3" />,
      label: basename(folders.find((f) => f.folder_id === filters.folderId)?.path ?? `Folder ${filters.folderId}`),
      tone: 'sky',
      onRemove: () => set('folderId', null),
    });

  const toneClass: Record<string, string> = {
    indigo: 'border-indigo-400/30 bg-indigo-500/10 text-indigo-200',
    violet: 'border-violet-400/30 bg-violet-500/10 text-violet-200',
    fuchsia: 'border-fuchsia-400/30 bg-fuchsia-500/10 text-fuchsia-200',
    pink: 'border-pink-400/30 bg-pink-500/10 text-pink-200',
    amber: 'border-amber-400/30 bg-amber-500/10 text-amber-200',
    emerald: 'border-emerald-400/30 bg-emerald-500/10 text-emerald-200',
    sky: 'border-sky-400/30 bg-sky-500/10 text-sky-200',
  };

  const data = searchQuery.data;
  const showResults = (Boolean(upload) || active) && data;
  const weights = filters.weights ?? DEFAULT_WEIGHTS;
  const setWeight = (k: keyof typeof DEFAULT_WEIGHTS, v: number) => set('weights', { ...weights, [k]: v });

  return (
    <div
      className="relative flex min-h-0 flex-1"
      onDragEnter={(e) => {
        if (![...e.dataTransfer.types].includes('Files')) return;
        dragDepth.current++;
        setDragging(true);
      }}
      onDragOver={(e) => {
        if ([...e.dataTransfer.types].includes('Files')) e.preventDefault();
      }}
      onDragLeave={() => {
        dragDepth.current = Math.max(0, dragDepth.current - 1);
        if (dragDepth.current === 0) setDragging(false);
      }}
      onDrop={(e) => {
        e.preventDefault();
        dragDepth.current = 0;
        setDragging(false);
        acceptFile(e.dataTransfer.files?.[0]);
      }}
    >
      {/* ---------- main column ---------- */}
      <div className="flex min-w-0 flex-1 flex-col overflow-y-auto">
        <div className="sticky top-0 z-10 border-b border-slate-800/60 bg-slate-950/80 px-5 pb-3 pt-4 backdrop-blur-md">
          {/* query bar */}
          <div className="flex items-center gap-2">
            <div className="group relative flex flex-1 items-center rounded-2xl border border-slate-700/70 bg-slate-900/80 shadow-[0_0_0_1px_rgba(99,102,241,0)] transition focus-within:border-indigo-500/70 focus-within:shadow-[0_0_30px_-6px_rgba(99,102,241,0.55)]">
              <Search className="ml-4 h-5 w-5 shrink-0 text-slate-500 group-focus-within:text-indigo-300" />
              <input
                ref={inputRef}
                value={filters.text}
                onChange={(e) => set('text', e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Escape') {
                    if (filters.text) set('text', '');
                    else (e.target as HTMLInputElement).blur();
                  }
                }}
                placeholder="Describe a vibe… “neon city at night”, “whiteboard with equations”"
                aria-label="Search by description"
                className="h-12 min-w-0 flex-1 bg-transparent px-3 text-[15px] text-slate-100 placeholder:text-slate-500 focus:outline-none"
              />
              {searchQuery.isFetching && <Spinner className="mr-2 h-4 w-4" />}
              {filters.text && (
                <button onClick={() => set('text', '')} className="btn-ghost mr-1" aria-label="Clear text">
                  <X className="h-4 w-4" />
                </button>
              )}
              <kbd className="mr-3 hidden rounded border border-slate-700 px-1.5 py-0.5 font-mono text-[10px] text-slate-500 xl:block">/</kbd>
            </div>
            <button
              onClick={() => fileRef.current?.click()}
              className="btn-secondary h-12 rounded-2xl px-4"
              title="Search by image (or drop / paste an image anywhere)"
            >
              <ImagePlus className="h-4 w-4" />
              <span className="hidden xl:inline">Image</span>
            </button>
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => {
                acceptFile(e.target.files?.[0]);
                e.target.value = '';
              }}
            />
            <button
              onClick={() => setPanelOpen((v) => !v)}
              aria-pressed={panelOpen}
              className={`btn-secondary h-12 rounded-2xl px-4 ${panelOpen ? '!border-indigo-500/50 !bg-indigo-500/10' : ''}`}
              title="Toggle sketch & filters panel"
            >
              <SlidersHorizontal className="h-4 w-4" />
            </button>
          </div>

          {/* chips + timing */}
          <div className="mt-2.5 flex min-h-[26px] flex-wrap items-center gap-1.5">
            {chips.length === 0 ? (
              <span className="text-xs text-slate-500">
                Modes: <span className="text-indigo-300">vibe</span> · <span className="text-violet-300">image</span> ·{' '}
                <span className="text-fuchsia-300">sketch</span> · <span className="text-pink-300">colour</span> ·{' '}
                <span className="text-amber-300">text-in-image</span> — combine any of them.
              </span>
            ) : (
              chips.map((c) => (
                <span key={c.key} className={`chip ${toneClass[c.tone]}`}>
                  {c.icon}
                  <span className="max-w-[220px] truncate">{c.label}</span>
                  <button onClick={c.onRemove} className="-mr-1 rounded-full p-0.5 hover:bg-white/10" aria-label="Remove">
                    <X className="h-3 w-3" />
                  </button>
                </span>
              ))
            )}
            {chips.length > 1 && (
              <button onClick={resetAll} className="btn-ghost py-0.5 text-[11px]">
                <RotateCcw className="h-3 w-3" /> Reset
              </button>
            )}
            {showResults && (
              <span className="ml-auto flex items-center gap-2 text-[11px] text-slate-400">
                {data.modes?.length > 0 && (
                  <span className="hidden text-slate-500 lg:inline">modes: {data.modes.join(' + ')}</span>
                )}
                <span className="flex items-center gap-1 rounded-full border border-slate-700/80 bg-slate-900 px-2 py-0.5 font-mono">
                  <Timer className="h-3 w-3 text-emerald-400" />
                  {formatMs(data.took_ms)}
                  <span className="text-slate-600">·</span>
                  <span className="text-slate-500">faiss</span> {formatMs(data.faiss_ms)}
                </span>
              </span>
            )}
          </div>
          {upload && chips.length > 1 && (
            <p className="mt-1.5 text-[11px] text-violet-300/80">
              Uploaded-image queries search by the image alone — remove it to combine the other filters.
            </p>
          )}
        </div>

        {/* results */}
        <div className="flex flex-1 flex-col px-5 py-4">
          {upload && (
            <div className="mb-4 flex items-center gap-3 rounded-xl border border-violet-500/20 bg-violet-500/5 p-2 pr-3">
              <img src={upload.previewUrl} alt="Query" className="h-14 w-14 rounded-lg object-cover" />
              <div className="min-w-0 flex-1 text-xs">
                <p className="font-medium text-violet-200">Searching by image</p>
                <p className="truncate text-slate-400">{upload.file.name || 'Pasted image'}</p>
              </div>
              <button onClick={clearUpload} className="btn-ghost">
                <X className="h-4 w-4" />
              </button>
            </div>
          )}

          {!upload && !active ? (
            <EmptyState icon={<Sparkles className="h-6 w-6" />} title="Search your images by vibe, sketch, colour or text">
              <p>Type a description, draw on the sketch-pad, pick colours, or drop an image here.</p>
              <div className="mt-4 flex flex-wrap justify-center gap-2">
                {EXAMPLES.map((ex) => (
                  <button
                    key={ex}
                    onClick={() => set('text', ex)}
                    className="chip border-slate-700 bg-slate-900 text-slate-300 hover:border-indigo-500/60 hover:text-indigo-200"
                  >
                    {ex}
                  </button>
                ))}
              </div>
            </EmptyState>
          ) : searchQuery.isError && !data ? (
            <ErrorState error={searchQuery.error} onRetry={() => searchQuery.refetch()} />
          ) : !data ? (
            <CenteredSpinner label="Searching…" />
          ) : data.results.length === 0 ? (
            <EmptyState icon={<Search className="h-6 w-6" />} title="No matches">
              Nothing matched these filters{data.total_candidates ? ` (${data.total_candidates} candidates scanned)` : ''}. Try
              loosening a filter or a different description.
            </EmptyState>
          ) : (
            <>
              {searchQuery.isError && (
                <div className="mb-3">
                  <ErrorState error={searchQuery.error} compact />
                </div>
              )}
              <div className="mb-3 flex items-center gap-3 text-xs text-slate-400">
                <span>
                  <span className="font-semibold text-slate-200">{data.results.length}</span> results
                  {data.total_candidates ? <span className="text-slate-500"> of {data.total_candidates} candidates</span> : null}
                </span>
                <span className="ml-auto hidden items-center gap-3 md:flex">
                  {(Object.keys(SCORE_COLORS) as (keyof typeof SCORE_COLORS)[]).map((k) => (
                    <span key={k} className="flex items-center gap-1 text-[10px] text-slate-500">
                      <span className="h-1.5 w-3 rounded-full" style={{ backgroundColor: SCORE_COLORS[k] }} /> {k}
                    </span>
                  ))}
                </span>
              </div>
              <div
                className={`grid grid-cols-[repeat(auto-fill,minmax(160px,1fr))] gap-3 transition-opacity ${
                  searchQuery.isPlaceholderData ? 'opacity-60' : ''
                }`}
              >
                {data.results.map((hit, i) => (
                  <ResultCard
                    key={hit.image_id}
                    hit={hit}
                    rank={i + 1}
                    isSelected={hit.image_id === selectedImageId}
                    onClick={() => onSelectImage(hit.image_id)}
                    clusterLabel={hit.cluster_id != null ? clusterMap.get(hit.cluster_id) : null}
                  />
                ))}
              </div>
            </>
          )}
        </div>
      </div>

      {/* ---------- right panel: sketch + filters ---------- */}
      {panelOpen && (
        <aside className="flex w-[300px] shrink-0 flex-col gap-3 overflow-y-auto border-l border-slate-800/60 bg-slate-950/40 p-3 xl:w-[320px]">
          <Section
            title="Live sketch-pad"
            icon={<PenTool className="h-3.5 w-3.5" />}
            open={sketchOpen}
            onToggle={() => setSketchOpen((v) => !v)}
            badge={filters.sketchPng ? <span className="chip border-fuchsia-400/30 bg-fuchsia-500/10 py-0 text-fuchsia-200">active</span> : null}
          >
            <SketchPad
              initialPng={filters.sketchPng}
              includeColors={filters.includeSketchColors}
              extractedColors={filters.sketchColors}
              onToggleIncludeColors={(v) => set('includeSketchColors', v)}
              onChange={(png, colors) => setFilters((f) => ({ ...f, sketchPng: png, sketchColors: colors }))}
              busy={searchQuery.isFetching && Boolean(filters.sketchPng)}
            />
          </Section>

          <Section
            title="Filters"
            icon={<SlidersHorizontal className="h-3.5 w-3.5" />}
            open={filtersOpen}
            onToggle={() => setFiltersOpen((v) => !v)}
          >
            <div className="space-y-4">
              <div>
                <label className="field-label" htmlFor="ocr-filter">
                  Text in image (OCR)
                </label>
                <div className="relative">
                  <Type className="pointer-events-none absolute left-2.5 top-2 h-4 w-4 text-slate-500" />
                  <input
                    id="ocr-filter"
                    value={filters.ocr}
                    onChange={(e) => set('ocr', e.target.value)}
                    placeholder="e.g. gradient descent"
                    className="input pl-8"
                  />
                </div>
              </div>

              <ColorPicker colors={filters.colors} onChange={(c) => set('colors', c)} />

              <div>
                <div className="field-label">
                  <span>Clusters</span>
                  {filters.clusterIds.length > 0 && (
                    <button onClick={() => set('clusterIds', [])} className="normal-case tracking-normal text-indigo-300 hover:underline">
                      clear
                    </button>
                  )}
                </div>
                {galaxyQuery.isLoading ? (
                  <Spinner className="h-4 w-4" />
                ) : clusters.length === 0 ? (
                  <p className="text-xs text-slate-500">
                    {galaxyQuery.isError ? 'No galaxy computed yet — compute one in the Galaxy view to filter by theme.' : 'No clusters.'}
                  </p>
                ) : (
                  <div className="flex max-h-36 flex-wrap gap-1.5 overflow-y-auto pr-1">
                    {clusters.map((c) => {
                      const on = filters.clusterIds.includes(c.cluster_id);
                      const col = clusterColor(c.color, c.cluster_id);
                      return (
                        <button
                          key={c.cluster_id}
                          onClick={() =>
                            set('clusterIds', on ? filters.clusterIds.filter((x) => x !== c.cluster_id) : [...filters.clusterIds, c.cluster_id])
                          }
                          aria-pressed={on}
                          className={`chip ${on ? 'border-transparent text-slate-950' : 'border-slate-700 bg-slate-900 text-slate-300 hover:border-slate-500'}`}
                          style={on ? { backgroundColor: col } : undefined}
                        >
                          {!on && <span className="h-2 w-2 rounded-full" style={{ backgroundColor: col }} />}
                          <span className="max-w-[140px] truncate">{c.label || `Cluster ${c.cluster_id}`}</span>
                          <span className="opacity-60">{c.size}</span>
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>

              <div>
                <label className="field-label" htmlFor="folder-filter">
                  Folder
                </label>
                <select
                  id="folder-filter"
                  value={filters.folderId ?? ''}
                  onChange={(e) => set('folderId', e.target.value === '' ? null : Number(e.target.value))}
                  className="input text-xs"
                >
                  <option value="">All folders</option>
                  {folders.map((f) => (
                    <option key={f.folder_id} value={f.folder_id}>
                      {basename(f.path)} ({f.image_counts?.total ?? 0})
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <div className="field-label">
                  <span>Fusion weights</span>
                  {filters.weights ? (
                    <button onClick={() => set('weights', null)} className="normal-case tracking-normal text-indigo-300 hover:underline">
                      auto
                    </button>
                  ) : (
                    <span className="normal-case tracking-normal text-slate-600">auto</span>
                  )}
                </div>
                <div className="space-y-2">
                  <Slider label="Semantic" accent={SCORE_COLORS.semantic} value={weights.semantic} min={0} max={1} step={0.05} onChange={(v) => setWeight('semantic', v)} format={(v) => v.toFixed(2)} />
                  <Slider label="OCR" accent={SCORE_COLORS.ocr} value={weights.ocr} min={0} max={1} step={0.05} onChange={(v) => setWeight('ocr', v)} format={(v) => v.toFixed(2)} />
                  <Slider label="Colour" accent={SCORE_COLORS.color} value={weights.color} min={0} max={1} step={0.05} onChange={(v) => setWeight('color', v)} format={(v) => v.toFixed(2)} />
                </div>
              </div>

              <Slider label="Results (k)" value={filters.k} min={12} max={200} step={4} onChange={(v) => set('k', v)} />
            </div>
          </Section>
        </aside>
      )}

      {/* drop overlay */}
      {dragging && (
        <div className="pointer-events-none absolute inset-3 z-30 flex flex-col items-center justify-center rounded-3xl border-2 border-dashed border-indigo-400/70 bg-indigo-950/70 backdrop-blur-sm">
          <ImageIcon className="mb-2 h-10 w-10 text-indigo-300" />
          <p className="text-sm font-medium text-indigo-100">Drop an image to find similar pictures</p>
        </div>
      )}
    </div>
  );
}

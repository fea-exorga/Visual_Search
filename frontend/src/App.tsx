import React, { useCallback, useEffect, useRef, useState } from 'react';
import Sidebar from './components/Sidebar';
import CenterPanel from './components/CenterPanel';
import InspectorDrawer from './components/InspectorDrawer';
import TopNav, { VIEWS } from './components/TopNav';
import SearchView from './components/search/SearchView';
import GalaxyView from './components/galaxy/GalaxyView';
import DuplicatesView from './components/duplicates/DuplicatesView';
import { DEFAULT_FILTERS, type SearchFilters, type UploadQuery } from './components/search/searchState';
import { useIndexing } from './hooks/useIndexing';
import { isTypingTarget } from './lib/format';
import type { ProjectionDims, ProjectionMethod, ViewName } from './types';

const VIEW_KEY = 'pixelindex.view';

function initialView(): ViewName {
  try {
    const hash = window.location.hash.replace('#', '') as ViewName;
    if (VIEWS.some((v) => v.id === hash)) return hash;
    const saved = localStorage.getItem(VIEW_KEY) as ViewName | null;
    if (saved && VIEWS.some((v) => v.id === saved)) return saved;
  } catch {
    /* ignore */
  }
  return 'search';
}

export default function App() {
  const [view, setView] = useState<ViewName>(initialView);
  const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth >= 1280);
  const [libraryFolderId, setLibraryFolderId] = useState<number | null>(null);
  const [selectedImageId, setSelectedImageId] = useState<number | null>(null);

  const [filters, setFilters] = useState<SearchFilters>(DEFAULT_FILTERS);
  const [upload, setUpload] = useState<UploadQuery | null>(null);
  const [projection, setProjection] = useState<{ method: ProjectionMethod; dims: ProjectionDims }>({ method: 'umap', dims: 2 });
  const [galaxyFocus, setGalaxyFocus] = useState<{ imageId: number; nonce: number } | null>(null);
  const focusNonce = useRef(1);
  const searchInputRef = useRef<HTMLInputElement>(null);

  const indexing = useIndexing();

  const changeView = useCallback((v: ViewName) => {
    setView(v);
    try {
      history.replaceState(null, '', `#${v}`);
      localStorage.setItem(VIEW_KEY, v);
    } catch {
      /* ignore */
    }
  }, []);

  const findSimilar = useCallback(
    (imageId: number) => {
      setUpload(null);
      setFilters((f) => ({ ...DEFAULT_FILTERS, k: f.k, includeSketchColors: f.includeSketchColors, imageId }));
      changeView('search');
    },
    [changeView],
  );

  const showInGalaxy = useCallback(
    (imageId: number) => {
      setGalaxyFocus({ imageId, nonce: focusNonce.current++ });
      changeView('galaxy');
    },
    [changeView],
  );

  // back/forward + manual hash edits
  useEffect(() => {
    const onHash = () => {
      const v = window.location.hash.replace('#', '') as ViewName;
      if (VIEWS.some((x) => x.id === v)) setView(v);
    };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  // global keyboard shortcuts
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === 'Escape' && selectedImageId !== null && !isTypingTarget(e.target)) {
        setSelectedImageId(null);
        return;
      }
      if (isTypingTarget(e.target)) return;
      if (e.key === '/') {
        e.preventDefault();
        changeView('search');
        window.setTimeout(() => searchInputRef.current?.focus(), 0);
      } else if (/^[1-4]$/.test(e.key) && !(e.target as HTMLElement)?.closest?.('canvas')) {
        changeView(VIEWS[Number(e.key) - 1].id);
      } else if (e.key === '[') {
        setSidebarOpen((v) => !v);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [changeView, selectedImageId]);

  return (
    <div className="galaxy-bg flex h-screen w-screen overflow-hidden text-slate-100">
      <div className="flex min-w-0 flex-1 flex-col">
        <TopNav view={view} onChangeView={changeView} onToggleSidebar={() => setSidebarOpen((v) => !v)} indexing={indexing} />
        <div className="flex min-h-0 flex-1">
          <Sidebar
            collapsed={!sidebarOpen}
            selectedFolderId={libraryFolderId}
            onSelectFolder={(id) => {
              setLibraryFolderId(id);
              changeView('library');
            }}
            indexing={indexing}
          />
          <main className="relative flex min-h-0 min-w-0 flex-1" role="tabpanel" aria-label={view}>
            {view === 'search' && (
              <SearchView
                filters={filters}
                setFilters={setFilters}
                upload={upload}
                setUpload={setUpload}
                projection={projection}
                selectedImageId={selectedImageId}
                onSelectImage={setSelectedImageId}
                inputRef={searchInputRef}
              />
            )}
            {view === 'galaxy' && (
              <GalaxyView
                method={projection.method}
                dims={projection.dims}
                onChangeProjection={(method, dims) => setProjection({ method, dims })}
                focus={galaxyFocus}
                onSelectImage={setSelectedImageId}
              />
            )}
            {view === 'duplicates' && <DuplicatesView selectedImageId={selectedImageId} onSelectImage={setSelectedImageId} />}
            {view === 'library' && (
              <CenterPanel
                folderId={libraryFolderId}
                selectedImageId={selectedImageId}
                onSelectImage={setSelectedImageId}
                indexing={indexing}
              />
            )}

            <InspectorDrawer
              imageId={selectedImageId}
              onClose={() => setSelectedImageId(null)}
              onFindSimilar={findSimilar}
              onShowInGalaxy={showInGalaxy}
            />
          </main>
        </div>
      </div>
    </div>
  );
}

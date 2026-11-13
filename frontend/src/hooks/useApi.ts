import type {
  Folder,
  ImageDetail,
  Stats,
  ImagesResponse,
  StartIndexResponse,
  AddFolderResponse,
  SystemInfo,
  SearchRequest,
  SearchResponse,
  RebuildResponse,
  GalaxyResponse,
  GalaxyComputeRequest,
  ProjectionMethod,
  ProjectionDims,
  DuplicatesResponse,
} from '../types';

export const BASE = '/api/v1';

/** Error carrying the HTTP status (0 = network failure / backend unreachable). */
export class ApiError extends Error {
  status: number;
  detail: string;
  constructor(status: number, detail: string) {
    super(status === 0 ? `Network error: ${detail}` : `API ${status}: ${detail}`);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
  /** Backend not reachable (fetch failed, or the Vite proxy answered 502/503/504/500-with-empty-body). */
  get isUnreachable(): boolean {
    return this.status === 0 || this.status === 502 || this.status === 503 || this.status === 504 || (this.status === 500 && !this.detail);
  }
  get isNotFound(): boolean {
    return this.status === 404;
  }
}

async function parseDetail(res: Response): Promise<string> {
  const text = await res.text().catch(() => '');
  if (!text) return '';
  try {
    const j = JSON.parse(text);
    if (typeof j?.detail === 'string') return j.detail;
    if (j?.detail) return JSON.stringify(j.detail);
    return text;
  } catch {
    return text.slice(0, 300);
  }
}

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  let res: Response;
  const isForm = options?.body instanceof FormData;
  try {
    res = await fetch(url, {
      ...options,
      headers: isForm ? options?.headers : { 'Content-Type': 'application/json', ...(options?.headers ?? {}) },
    });
  } catch (e) {
    if ((e as Error)?.name === 'AbortError') throw e;
    throw new ApiError(0, (e as Error)?.message || 'fetch failed');
  }
  if (!res.ok) {
    throw new ApiError(res.status, await parseDetail(res));
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

/** Human-readable error description for empty/error states. */
export function describeError(err: unknown): { title: string; detail: string; kind: 'down' | 'missing' | 'other' } {
  if (err instanceof ApiError) {
    if (err.isUnreachable) {
      return {
        kind: 'down',
        title: 'Backend unreachable',
        detail: 'Could not reach the PixelIndex API on localhost:8000. Start it with `uvicorn app.main:app`.',
      };
    }
    if (err.isNotFound || err.status === 405) {
      return {
        kind: 'missing',
        title: 'Endpoint not available',
        detail: err.detail && err.detail !== 'Not Found' ? err.detail : 'This backend does not implement this endpoint yet.',
      };
    }
    return { kind: 'other', title: `Request failed (${err.status})`, detail: err.detail || err.message };
  }
  return { kind: 'other', title: 'Something went wrong', detail: (err as Error)?.message ?? String(err) };
}

// ---- Folders ----
export function getFolders(): Promise<Folder[]> {
  return request<Folder[]>(`${BASE}/folders`);
}

export function addFolder(path: string): Promise<AddFolderResponse> {
  return request<AddFolderResponse>(`${BASE}/folders`, {
    method: 'POST',
    body: JSON.stringify({ path }),
  });
}

export function deleteFolder(folderId: number): Promise<void> {
  return request<void>(`${BASE}/folders/${folderId}`, { method: 'DELETE' });
}

// ---- Indexing ----
export function startIndex(folderIds: number[], forceReindex = false): Promise<StartIndexResponse> {
  return request<StartIndexResponse>(`${BASE}/index/start`, {
    method: 'POST',
    body: JSON.stringify({ folder_ids: folderIds, force_reindex: forceReindex }),
  });
}

export function pauseIndex(jobId: string): Promise<void> {
  return request<void>(`${BASE}/index/pause`, {
    method: 'POST',
    body: JSON.stringify({ job_id: jobId }),
  });
}

export function resumeIndex(jobId: string): Promise<void> {
  return request<void>(`${BASE}/index/resume`, {
    method: 'POST',
    body: JSON.stringify({ job_id: jobId }),
  });
}

// ---- Images ----
export function getImages(params: {
  limit?: number;
  offset?: number;
  folder_id?: number;
  status?: string;
}): Promise<ImagesResponse> {
  const qs = new URLSearchParams();
  if (params.limit !== undefined) qs.set('limit', String(params.limit));
  if (params.offset !== undefined) qs.set('offset', String(params.offset));
  if (params.folder_id !== undefined) qs.set('folder_id', String(params.folder_id));
  if (params.status) qs.set('status', params.status);
  return request<ImagesResponse>(`${BASE}/images?${qs}`);
}

export function getImageDetail(imageId: number): Promise<ImageDetail> {
  return request<ImageDetail>(`${BASE}/images/${imageId}`);
}

export function thumbnailUrl(imageId: number): string {
  return `${BASE}/thumbnails/${imageId}`;
}

// ---- Stats / system ----
export function getStats(): Promise<Stats> {
  return request<Stats>(`${BASE}/stats`);
}

export function getSystemInfo(): Promise<SystemInfo> {
  return request<SystemInfo>(`${BASE}/system/info`);
}

// ---- Search (M2) ----
export function search(body: SearchRequest, signal?: AbortSignal): Promise<SearchResponse> {
  return request<SearchResponse>(`${BASE}/search`, {
    method: 'POST',
    body: JSON.stringify(body),
    signal,
  });
}

export function searchUpload(file: Blob, k?: number, signal?: AbortSignal): Promise<SearchResponse> {
  const fd = new FormData();
  fd.append('file', file, (file as File).name || 'upload.png');
  if (k !== undefined) fd.append('k', String(k));
  return request<SearchResponse>(`${BASE}/search/upload`, { method: 'POST', body: fd, signal });
}

export function searchSimilar(imageId: number, k = 24, signal?: AbortSignal): Promise<SearchResponse> {
  return request<SearchResponse>(`${BASE}/search/similar/${imageId}?k=${k}`, { signal });
}

export function rebuildSearchIndex(): Promise<RebuildResponse> {
  return request<RebuildResponse>(`${BASE}/search/rebuild`, { method: 'POST' });
}

// ---- Galaxy (M3) ----
export function getGalaxy(method: ProjectionMethod, dims: ProjectionDims, signal?: AbortSignal): Promise<GalaxyResponse> {
  return request<GalaxyResponse>(`${BASE}/galaxy?method=${method}&dims=${dims}`, { signal });
}

export function computeGalaxy(body: GalaxyComputeRequest): Promise<GalaxyResponse> {
  return request<GalaxyResponse>(`${BASE}/galaxy/compute`, {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

// ---- Duplicates (M5) ----
export function getDuplicates(threshold = 0.95, maxHamming = 6, signal?: AbortSignal): Promise<DuplicatesResponse> {
  return request<DuplicatesResponse>(
    `${BASE}/duplicates?threshold=${threshold}&max_hamming=${maxHamming}`,
    { signal },
  );
}

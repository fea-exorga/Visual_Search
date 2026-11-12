import type {
  Folder,
  ImageSummary,
  ImageDetail,
  Stats,
  ImagesResponse,
  StartIndexResponse,
  AddFolderResponse,
} from '../types';

const BASE = '/api/v1';

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
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
export function startIndex(
  folderIds: number[],
  forceReindex = false
): Promise<StartIndexResponse> {
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

// ---- Stats ----
export function getStats(): Promise<Stats> {
  return request<Stats>(`${BASE}/stats`);
}

import type { ProjectionDims, ProjectionMethod, SearchRequest } from '../../types';

export interface Weights {
  semantic: number;
  ocr: number;
  color: number;
}

export const DEFAULT_WEIGHTS: Weights = { semantic: 1, ocr: 0.6, color: 0.6 };

export interface SearchFilters {
  text: string;
  ocr: string;
  colors: string[];
  clusterIds: number[];
  folderId: number | null;
  /** null = let the backend use its defaults. */
  weights: Weights | null;
  k: number;
  /** Query by an already-indexed image ("Find similar"). */
  imageId: number | null;
  sketchPng: string | null;
  sketchColors: string[];
  includeSketchColors: boolean;
}

export const DEFAULT_FILTERS: SearchFilters = {
  text: '',
  ocr: '',
  colors: [],
  clusterIds: [],
  folderId: null,
  weights: null,
  k: 60,
  imageId: null,
  sketchPng: null,
  sketchColors: [],
  includeSketchColors: true,
};

export interface UploadQuery {
  id: number;
  file: File;
  previewUrl: string;
}

export const MAX_COLORS = 5;

/** Colours actually sent: user swatches + (optionally) sketch colours, de-duplicated, max 5. */
export function effectiveColors(f: SearchFilters): string[] {
  const out: string[] = [];
  const push = (c: string) => {
    const u = c.toUpperCase();
    if (!out.includes(u) && out.length < MAX_COLORS) out.push(u);
  };
  f.colors.forEach(push);
  if (f.sketchPng && f.includeSketchColors) f.sketchColors.forEach(push);
  return out;
}

/** Whether the filters contain at least one query mode (filters alone don't make a query). */
export function hasQuery(f: SearchFilters, text: string): boolean {
  return Boolean(text.trim() || f.ocr.trim() || f.imageId != null || f.sketchPng || effectiveColors(f).length);
}

export function buildSearchRequest(
  f: SearchFilters,
  text: string,
  projection: { method: ProjectionMethod; dims: ProjectionDims },
): SearchRequest {
  // projection is always sent so hits come back with their cluster_id (used for labels)
  const req: SearchRequest = { k: f.k, projection };
  if (text.trim()) req.text = text.trim();
  if (f.ocr.trim()) req.ocr = f.ocr.trim();
  if (f.imageId != null) req.image_id = f.imageId;
  if (f.sketchPng) req.sketch_png = f.sketchPng;
  const colors = effectiveColors(f);
  if (colors.length) req.colors = colors;
  if (f.clusterIds.length) {
    req.cluster_ids = [...f.clusterIds].sort((a, b) => a - b);
  }
  if (f.folderId != null) req.folder_id = f.folderId;
  if (f.weights) req.weights = { ...f.weights };
  return req;
}

/** "Find similar" can use the dedicated GET endpoint when nothing else is combined with it. */
export function isPureSimilar(f: SearchFilters, text: string): boolean {
  return (
    f.imageId != null &&
    !text.trim() &&
    !f.ocr.trim() &&
    !f.sketchPng &&
    effectiveColors(f).length === 0 &&
    f.clusterIds.length === 0 &&
    f.folderId == null &&
    f.weights == null
  );
}

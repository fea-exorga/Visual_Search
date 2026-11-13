// ---- Folders ----
export interface FolderImageCounts {
  total: number;
  completed: number;
  pending: number;
  error: number;
}

/** GET /folders row. `enabled` is 0/1 from SQLite (soft-deleted folders have 0). */
export interface Folder {
  folder_id: number;
  path: string;
  enabled: boolean | number;
  last_scan_at: string | null;
  created_at: string | null;
  image_counts: FolderImageCounts;
}

// ---- Colors ----
export interface ColorSwatch {
  rank: number;
  hex_code: string;
  r: number;
  g: number;
  b: number;
  weight: number;
}

// ---- OCR ----
export interface OCRResult {
  raw_text: string;
  mean_confidence: number;
  word_count: number;
}

// ---- Images ----
export interface ImageSummary {
  image_id: number;
  file_path: string;
  thumbnail_url: string;
  width: number;
  height: number;
  status: string;
  colors: ColorSwatch[];
  ocr_snippet: string | null;
}

export interface ImageDetail {
  image_id: number;
  folder_id?: number;
  file_path: string;
  sha256: string;
  width: number;
  height: number;
  file_size: number;
  format: string;
  status: string;
  thumbnail_url?: string;
  has_embedding?: boolean;
  embedding_dim: number | null;
  model_name: string | null;
  l2_norm: number | null;
  colors: ColorSwatch[];
  ocr: OCRResult | null;
}

// ---- Index Progress (SSE) ----
export interface IndexProgress {
  scanned: number;
  color_done: number;
  ocr_done: number;
  embedded: number;
  skipped: number;
  failed: number;
  total: number;
  stage: string;
  eta_sec: number | null;
  images_per_sec: number | null;
}

// ---- Stats ----
export interface Stats {
  total_images: number;
  by_status: Record<string, number>;
  total_folders: number;
  disk_usage_bytes: number;
}

// ---- System ----
export interface SystemInfo {
  embed_backend: string;
  onnx_repo?: string;
  onnx_precision: string;
  embedding_dim: number;
  ocr_engine: string;
  vector_models: { model: string; count: number }[];
}

// ---- API responses ----
export interface ImagesResponse {
  images: ImageSummary[];
  total: number;
  limit?: number;
  offset?: number;
}

export interface StartIndexResponse {
  job_id: string;
}

export type AddFolderResponse = Omit<Folder, 'image_counts'> & Partial<Pick<Folder, 'image_counts'>>;

// ---- Search (PRD §6.4, M2) ----
export type ProjectionMethod = 'umap' | 'tsne';
export type ProjectionDims = 2 | 3;

export interface SearchWeights {
  semantic?: number;
  ocr?: number;
  color?: number;
}

export interface SearchRequest {
  text?: string;
  image_id?: number;
  sketch_png?: string;
  colors?: string[];
  ocr?: string;
  cluster_ids?: number[];
  folder_id?: number;
  projection?: { method: ProjectionMethod; dims: ProjectionDims };
  k?: number;
  weights?: SearchWeights;
}

export interface SearchScores {
  semantic?: number | null;
  ocr?: number | null;
  color?: number | null;
}

export interface SearchHit {
  image_id: number;
  score: number;
  scores: SearchScores;
  file_path: string;
  thumbnail_url: string;
  width: number;
  height: number;
  colors: ColorSwatch[];
  ocr_snippet: string | null;
  cluster_id?: number | null;
}

export interface SearchResponse {
  results: SearchHit[];
  took_ms: number;
  faiss_ms: number;
  modes: string[];
  total_candidates: number;
}

export interface RebuildResponse {
  n_vectors: number;
  index_type: string;
  build_ms: number;
}

// ---- Galaxy (PRD §6.4, M3) ----
export interface GalaxyPoint {
  image_id: number;
  x: number;
  y: number;
  z?: number | null;
  cluster_id: number;
}

export interface GalaxyCluster {
  cluster_id: number;
  label: string;
  size: number;
  color: string;
  centroid: number[];
}

export interface GalaxyResponse {
  method: ProjectionMethod;
  dims: ProjectionDims;
  points: GalaxyPoint[];
  clusters: GalaxyCluster[];
  computed_at: string | number | null;
  /** Optional: true when images were indexed after the projection was computed. */
  stale?: boolean;
}

export interface GalaxyComputeRequest {
  method: ProjectionMethod;
  dims: ProjectionDims;
  n_neighbors?: number;
  min_dist?: number;
  perplexity?: number;
  n_clusters?: number;
}

// ---- Duplicates (PRD §6.4, M5) ----
export interface DuplicateImage {
  image_id: number;
  file_path: string;
  thumbnail_url: string;
  file_size: number;
  width: number;
  height: number;
  similarity: number;
}

export interface DuplicateGroup {
  group_id: number;
  kind: 'exact' | 'near';
  images: DuplicateImage[];
}

export interface DuplicatesResponse {
  groups: DuplicateGroup[];
  n_groups: number;
  n_images: number;
}

// ---- App navigation ----
export type ViewName = 'search' | 'galaxy' | 'duplicates' | 'library';

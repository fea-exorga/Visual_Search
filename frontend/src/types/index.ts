// ---- Folders ----
export interface Folder {
  folder_id: number;
  path: string;
  enabled: boolean;
  total_images: number;
  completed: number;
  pending: number;
  error_count: number;
  last_scan_at: string | null;
  image_count?: number;
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
  file_path: string;
  sha256: string;
  width: number;
  height: number;
  file_size: number;
  format: string;
  status: string;
  embedding_row_id: number | null;
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

// ---- API responses ----
export interface ImagesResponse {
  images: ImageSummary[];
  total: number;
}

export interface StartIndexResponse {
  job_id: string;
}

export interface AddFolderResponse {
  folder_id: number;
  path: string;
  enabled: boolean;
  image_count: number;
}

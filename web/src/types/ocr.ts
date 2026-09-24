export type OcrModel = "paddleocr" | "got-ocr2";

export type MediaKind = "image" | "video";

/** Normalized (0-1) bounding box plus recognized text, matching the pipeline's JSON output shape. */
export interface OcrBox {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  text: string;
  confidence: number;
  model: OcrModel;
  timestamp?: number;
  /** LLM-corrected version of `text` (spelling/OCR-error fixes only), when the cleanup stage is enabled. */
  cleanedText?: string;
}

export interface OcrResult {
  boxes: OcrBox[];
}

export type PipelineStageId = "frame-sampling" | "detection" | "recognition" | "merging";

export type StageStatus = "pending" | "running" | "done";

export interface PipelineStage {
  id: PipelineStageId;
  label: string;
  status: StageStatus;
  progress: number;
}

export type AppPhase = "upload" | "processing" | "results";

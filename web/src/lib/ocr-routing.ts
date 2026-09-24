import type { RawDetection } from "@/lib/mock-fixtures";
import type { OcrBox, OcrResult } from "@/types/ocr";

/** Confidence boost applied when GOT-OCR2.0 re-reads a crop the fast path was unsure about. */
const SPECIALIST_CONFIDENCE_BOOST = 0.28;
const SPECIALIST_CONFIDENCE_CAP = 0.99;

export function escalates(detection: RawDetection, threshold: number): boolean {
  return detection.rawConfidence < threshold;
}

/**
 * Resolves raw PaddleOCR detections into the final routed result, applying the confidence
 * threshold live so the settings panel's slider can be reflected without re-running the pipeline.
 */
export function routeDetections(
  detections: RawDetection[],
  threshold: number,
  videoDurationSeconds?: number,
): OcrResult {
  const boxes: OcrBox[] = detections.map((detection) => {
    const escalate = escalates(detection, threshold);
    const model = escalate ? "got-ocr2" : "paddleocr";
    const text =
      escalate && detection.specialistText ? detection.specialistText : detection.candidateText;
    const confidence = escalate
      ? Math.min(SPECIALIST_CONFIDENCE_CAP, detection.rawConfidence + SPECIALIST_CONFIDENCE_BOOST)
      : detection.rawConfidence;

    const cleanedText = escalate
      ? detection.cleanedSpecialistText
      : detection.cleanedCandidateText;

    const box: OcrBox = {
      id: detection.id,
      x: detection.x,
      y: detection.y,
      w: detection.w,
      h: detection.h,
      text,
      confidence,
      model,
    };

    if (cleanedText !== undefined) {
      box.cleanedText = cleanedText;
    }

    if (detection.relativeTimestamp !== undefined) {
      box.timestamp =
        videoDurationSeconds !== undefined
          ? detection.relativeTimestamp * videoDurationSeconds
          : detection.relativeTimestamp;
    }

    return box;
  });

  return { boxes };
}

export interface RoutingStats {
  total: number;
  fastPath: number;
  specialist: number;
  specialistShare: number;
}

export function computeRoutingStats(detections: RawDetection[], threshold: number): RoutingStats {
  const total = detections.length;
  const specialist = detections.filter((detection) => escalates(detection, threshold)).length;
  return {
    total,
    specialist,
    fastPath: total - specialist,
    specialistShare: total === 0 ? 0 : specialist / total,
  };
}

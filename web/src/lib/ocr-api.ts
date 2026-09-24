import type { RawDetection } from "@/lib/mock-fixtures";

const API_BASE_URL = process.env["NEXT_PUBLIC_OCR_API_URL"] ?? "http://localhost:8000";

interface OcrApiResponse {
  detections: RawDetection[];
}

export async function fetchDetections(file: File, signal: AbortSignal): Promise<RawDetection[]> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/api/ocr`, {
    method: "POST",
    body: formData,
    signal,
  });

  if (!response.ok) {
    throw new Error(`OCR backend returned ${response.status}`);
  }

  const data = (await response.json()) as OcrApiResponse;
  // The backend serializes an absent timestamp as JSON null; normalize to
  // undefined here so it matches RawDetection's declared type everywhere else.
  return data.detections.map((detection) => ({
    ...detection,
    specialistText: detection.specialistText ?? undefined,
    relativeTimestamp: detection.relativeTimestamp ?? undefined,
  }));
}

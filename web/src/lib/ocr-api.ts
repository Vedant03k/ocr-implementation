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
  // The backend serializes an absent optional field as JSON null, but
  // RawDetection's optional fields must be omitted entirely (not set to
  // undefined) under exactOptionalPropertyTypes, so drop nullish ones here.
  return data.detections.map((detection) => {
    const { specialistText, cleanedCandidateText, cleanedSpecialistText, relativeTimestamp, ...rest } =
      detection;
    return {
      ...rest,
      ...(specialistText != null ? { specialistText } : {}),
      ...(cleanedCandidateText != null ? { cleanedCandidateText } : {}),
      ...(cleanedSpecialistText != null ? { cleanedSpecialistText } : {}),
      ...(relativeTimestamp != null ? { relativeTimestamp } : {}),
    };
  });
}

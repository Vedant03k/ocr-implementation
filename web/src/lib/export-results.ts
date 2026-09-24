import type { OcrResult } from "@/types/ocr";

export function resultToJson(result: OcrResult): string {
  return JSON.stringify(result, null, 2);
}

export function resultToPlainText(result: OcrResult): string {
  const sorted = [...result.boxes].sort((a, b) => {
    if (a.timestamp !== undefined && b.timestamp !== undefined) return a.timestamp - b.timestamp;
    return a.y - b.y;
  });
  return sorted.map((box) => box.text).join("\n");
}

export function resultToSrt(result: OcrResult): string {
  const withTimestamps = result.boxes
    .filter((box): box is typeof box & { timestamp: number } => box.timestamp !== undefined)
    .sort((a, b) => a.timestamp - b.timestamp);

  return withTimestamps
    .map((box, index) => {
      const start = formatSrtTimestamp(box.timestamp);
      const end = formatSrtTimestamp(box.timestamp + 2.5);
      return `${index + 1}\n${start} --> ${end}\n${box.text}\n`;
    })
    .join("\n");
}

function formatSrtTimestamp(totalSeconds: number): string {
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = Math.floor(totalSeconds % 60);
  const milliseconds = Math.round((totalSeconds - Math.floor(totalSeconds)) * 1000);

  const pad = (value: number, length = 2): string => value.toString().padStart(length, "0");
  return `${pad(hours)}:${pad(minutes)}:${pad(seconds)},${pad(milliseconds, 3)}`;
}

export function downloadTextFile(
  filename: string,
  contents: string,
  mimeType = "text/plain",
): void {
  const blob = new Blob([contents], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

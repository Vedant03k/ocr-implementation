"use client";

import { Download } from "lucide-react";

import {
  downloadTextFile,
  resultToJson,
  resultToPlainText,
  resultToSrt,
} from "@/lib/export-results";
import type { MediaKind, OcrResult } from "@/types/ocr";

interface ExportButtonsProps {
  result: OcrResult;
  mediaKind: MediaKind;
}

export function ExportButtons({ result, mediaKind }: ExportButtonsProps) {
  return (
    <div className="flex flex-wrap gap-2">
      <ExportButton
        label="JSON"
        onClick={() =>
          downloadTextFile("ocr-results.json", resultToJson(result), "application/json")
        }
      />
      <ExportButton
        label="Plain text"
        onClick={() => downloadTextFile("ocr-results.txt", resultToPlainText(result), "text/plain")}
      />
      {mediaKind === "video" ? (
        <ExportButton
          label="SRT"
          onClick={() =>
            downloadTextFile("ocr-results.srt", resultToSrt(result), "application/x-subrip")
          }
        />
      ) : null}
    </div>
  );
}

function ExportButton({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="inline-flex items-center gap-1.5 rounded-md border border-zinc-200 px-3 py-1.5 text-xs font-medium text-zinc-700 transition-colors hover:border-zinc-300 hover:bg-zinc-50 dark:border-zinc-800 dark:text-zinc-300 dark:hover:border-zinc-700 dark:hover:bg-zinc-900"
    >
      <Download className="h-3.5 w-3.5" />
      {label}
    </button>
  );
}

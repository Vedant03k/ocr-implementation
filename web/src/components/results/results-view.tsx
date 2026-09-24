"use client";

import { RotateCcw } from "lucide-react";
import { useMemo, useState } from "react";

import { ExportButtons } from "@/components/results/export-buttons";
import { ImageResults } from "@/components/results/image-results";
import { ResultsTextPanel } from "@/components/results/results-text-panel";
import { VideoResults } from "@/components/results/video-results";
import { SettingsPanel } from "@/components/settings-panel";
import type { RawDetection } from "@/lib/mock-fixtures";
import { computeRoutingStats, routeDetections } from "@/lib/ocr-routing";
import type { UploadedMedia } from "@/hooks/use-ocr-pipeline";
import type { OcrResult } from "@/types/ocr";

interface ResultsViewProps {
  media: UploadedMedia;
  rawDetections: RawDetection[];
  confidenceThreshold: number;
  onThresholdChange: (value: number) => void;
  videoDurationSeconds: number | null;
  onVideoDurationLoaded: (seconds: number) => void;
  onReset: () => void;
}

export function ResultsView({
  media,
  rawDetections,
  confidenceThreshold,
  onThresholdChange,
  videoDurationSeconds,
  onVideoDurationLoaded,
  onReset,
}: ResultsViewProps) {
  const result = useMemo(
    () => routeDetections(rawDetections, confidenceThreshold, videoDurationSeconds ?? undefined),
    [rawDetections, confidenceThreshold, videoDurationSeconds],
  );

  const routingStats = useMemo(
    () => computeRoutingStats(rawDetections, confidenceThreshold),
    [rawDetections, confidenceThreshold],
  );

  const hasCleanedText = useMemo(
    () => result.boxes.some((box) => box.cleanedText !== undefined),
    [result],
  );
  const [showCleaned, setShowCleaned] = useState(true);

  // Swaps each box's displayed text for its LLM-corrected version (when available and
  // enabled), so every consumer below — overlay tooltips, box detail, export, copy — just
  // reads `text` as usual without needing to know cleanup exists.
  const displayResult: OcrResult = useMemo(() => {
    if (!showCleaned) return result;
    return {
      boxes: result.boxes.map((box) =>
        box.cleanedText !== undefined ? { ...box, text: box.cleanedText } : box,
      ),
    };
  }, [result, showCleaned]);

  return (
    <div className="grid w-full grid-cols-1 gap-6 lg:grid-cols-[1fr_300px]">
      <div className="flex flex-col gap-4">
        {media.mediaKind === "image" ? (
          <ImageResults previewUrl={media.previewUrl} boxes={displayResult.boxes} />
        ) : (
          <VideoResults
            previewUrl={media.previewUrl}
            boxes={displayResult.boxes}
            onDurationLoaded={onVideoDurationLoaded}
          />
        )}
      </div>

      <div className="flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <ExportButtons result={displayResult} mediaKind={media.mediaKind} />
          <button
            type="button"
            onClick={onReset}
            className="inline-flex shrink-0 items-center gap-1.5 rounded-md px-2 py-1.5 text-xs font-medium text-zinc-500 transition-colors hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            New file
          </button>
        </div>

        {hasCleanedText ? (
          <label className="flex items-center gap-2 text-xs text-zinc-600 dark:text-zinc-400">
            <input
              type="checkbox"
              checked={showCleaned}
              onChange={(event) => setShowCleaned(event.target.checked)}
              className="accent-zinc-900 dark:accent-zinc-100"
            />
            Show corrected text (spelling/OCR fixes)
          </label>
        ) : null}

        <SettingsPanel
          confidenceThreshold={confidenceThreshold}
          onThresholdChange={onThresholdChange}
          routingStats={routingStats}
        />

        <ResultsTextPanel result={displayResult} />
      </div>
    </div>
  );
}

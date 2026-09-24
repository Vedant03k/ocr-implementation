"use client";

import { RotateCcw } from "lucide-react";
import { useMemo } from "react";

import { ExportButtons } from "@/components/results/export-buttons";
import { ImageResults } from "@/components/results/image-results";
import { ResultsTextPanel } from "@/components/results/results-text-panel";
import { VideoResults } from "@/components/results/video-results";
import { SettingsPanel } from "@/components/settings-panel";
import type { RawDetection } from "@/lib/mock-fixtures";
import { computeRoutingStats, routeDetections } from "@/lib/ocr-routing";
import type { UploadedMedia } from "@/hooks/use-ocr-pipeline";

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

  return (
    <div className="grid w-full grid-cols-1 gap-6 lg:grid-cols-[1fr_300px]">
      <div className="flex flex-col gap-4">
        {media.mediaKind === "image" ? (
          <ImageResults previewUrl={media.previewUrl} boxes={result.boxes} />
        ) : (
          <VideoResults
            previewUrl={media.previewUrl}
            boxes={result.boxes}
            onDurationLoaded={onVideoDurationLoaded}
          />
        )}
      </div>

      <div className="flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <ExportButtons result={result} mediaKind={media.mediaKind} />
          <button
            type="button"
            onClick={onReset}
            className="inline-flex shrink-0 items-center gap-1.5 rounded-md px-2 py-1.5 text-xs font-medium text-zinc-500 transition-colors hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            New file
          </button>
        </div>

        <SettingsPanel
          confidenceThreshold={confidenceThreshold}
          onThresholdChange={onThresholdChange}
          routingStats={routingStats}
        />

        <ResultsTextPanel result={result} />
      </div>
    </div>
  );
}

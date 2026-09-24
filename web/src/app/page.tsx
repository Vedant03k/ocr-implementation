"use client";

import { ScanText } from "lucide-react";

import { ProcessingView } from "@/components/processing-view";
import { ResultsView } from "@/components/results/results-view";
import { ThemeToggle } from "@/components/theme-toggle";
import { UploadZone } from "@/components/upload-zone";
import { useOcrPipeline } from "@/hooks/use-ocr-pipeline";

export default function Home() {
  const { state, selectMedia, reset, setThreshold, setVideoDuration } = useOcrPipeline();
  const { phase, media, stages, confidenceThreshold, rawDetections, videoDurationSeconds, error } = state;

  return (
    <div className="flex min-h-full flex-1 flex-col bg-white dark:bg-zinc-950">
      <header className="flex items-center justify-between border-b border-zinc-200 px-6 py-4 dark:border-zinc-800">
        <div className="flex items-center gap-2">
          <ScanText className="h-5 w-5 text-zinc-900 dark:text-zinc-100" />
          <span className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
            OCR Pipeline
          </span>
        </div>
        <ThemeToggle />
      </header>

      <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col items-center justify-center px-6 py-12">
        {phase === "upload" ? (
          <div className="w-full max-w-lg">
            {error ? (
              <p className="mb-4 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300">
                {error}
              </p>
            ) : null}
            <UploadZone onFileSelected={selectMedia} />
          </div>
        ) : null}

        {phase === "processing" ? (
          <div className="w-full max-w-lg">
            <ProcessingView stages={stages} />
          </div>
        ) : null}

        {phase === "results" && media && rawDetections ? (
          <ResultsView
            media={media}
            rawDetections={rawDetections}
            confidenceThreshold={confidenceThreshold}
            onThresholdChange={setThreshold}
            videoDurationSeconds={videoDurationSeconds}
            onVideoDurationLoaded={setVideoDuration}
            onReset={reset}
          />
        ) : null}
      </main>
    </div>
  );
}

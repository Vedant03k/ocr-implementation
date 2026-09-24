"use client";

import { Check, Loader2 } from "lucide-react";

import type { RoutingStats } from "@/lib/ocr-routing";
import type { PipelineStage } from "@/types/ocr";

interface ProcessingViewProps {
  stages: PipelineStage[];
  routingStats: RoutingStats;
}

export function ProcessingView({ stages, routingStats }: ProcessingViewProps) {
  const recognitionStage = stages.find((stage) => stage.id === "recognition");

  return (
    <div className="w-full rounded-lg border border-zinc-200 p-6 dark:border-zinc-800">
      <ol className="flex flex-col gap-4">
        {stages.map((stage) => (
          <li key={stage.id}>
            <div className="flex items-center gap-3">
              <StageIcon status={stage.status} />
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <span
                    className={`text-sm font-medium ${
                      stage.status === "pending"
                        ? "text-zinc-400 dark:text-zinc-600"
                        : "text-zinc-900 dark:text-zinc-100"
                    }`}
                  >
                    {stage.label}
                  </span>
                  <span className="font-mono text-xs text-zinc-400 dark:text-zinc-600">
                    {stage.status === "pending" ? "" : `${stage.progress}%`}
                  </span>
                </div>
                <div className="mt-1.5 h-1 w-full overflow-hidden rounded-full bg-zinc-100 dark:bg-zinc-800">
                  <div
                    className="h-full bg-zinc-900 transition-[width] duration-150 dark:bg-zinc-100"
                    style={{ width: `${stage.progress}%` }}
                  />
                </div>
              </div>
            </div>

            {stage.id === "recognition" &&
            recognitionStage &&
            recognitionStage.status !== "pending" ? (
              <div className="mt-3 ml-8 grid grid-cols-2 gap-2">
                <RecognitionPathBar
                  label="Fast path"
                  sublabel="PaddleOCR"
                  count={routingStats.fastPath}
                  active={recognitionStage.status === "running"}
                  barClassName="bg-emerald-500"
                />
                <RecognitionPathBar
                  label="Specialist"
                  sublabel="GOT-OCR2.0"
                  count={routingStats.specialist}
                  active={recognitionStage.status === "running"}
                  barClassName="bg-amber-500"
                />
              </div>
            ) : null}
          </li>
        ))}
      </ol>
    </div>
  );
}

function StageIcon({ status }: { status: PipelineStage["status"] }) {
  if (status === "done") {
    return (
      <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-zinc-900 dark:bg-zinc-100">
        <Check className="h-3 w-3 text-white dark:text-zinc-900" />
      </span>
    );
  }
  if (status === "running") {
    return (
      <span className="flex h-5 w-5 shrink-0 items-center justify-center">
        <Loader2 className="h-4 w-4 animate-spin text-zinc-900 dark:text-zinc-100" />
      </span>
    );
  }
  return (
    <span className="h-5 w-5 shrink-0 rounded-full border border-zinc-200 dark:border-zinc-800" />
  );
}

interface RecognitionPathBarProps {
  label: string;
  sublabel: string;
  count: number;
  active: boolean;
  barClassName: string;
}

function RecognitionPathBar({
  label,
  sublabel,
  count,
  active,
  barClassName,
}: RecognitionPathBarProps) {
  return (
    <div className="rounded-md border border-zinc-200 p-2 dark:border-zinc-800">
      <div className="flex items-baseline justify-between">
        <span className="text-xs font-medium text-zinc-900 dark:text-zinc-100">{label}</span>
        <span className="font-mono text-xs text-zinc-500 dark:text-zinc-400">{count}</span>
      </div>
      <p className="text-[11px] text-zinc-500 dark:text-zinc-500">{sublabel}</p>
      {active ? (
        <div className="mt-1.5 h-0.5 w-full overflow-hidden rounded-full bg-zinc-100 dark:bg-zinc-800">
          <div className={`h-full w-full animate-pulse ${barClassName}`} />
        </div>
      ) : null}
    </div>
  );
}

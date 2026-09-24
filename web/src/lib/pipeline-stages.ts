import type { MediaKind, PipelineStage, PipelineStageId } from "@/types/ocr";

const STAGE_LABELS: Record<PipelineStageId, string> = {
  "frame-sampling": "Frame sampling",
  detection: "Detection",
  recognition: "Recognition",
  merging: "Merging",
};

/** Roughly how long each stage's simulated progress bar takes to fill, in ms. */
const STAGE_DURATIONS_MS: Record<PipelineStageId, number> = {
  "frame-sampling": 900,
  detection: 1100,
  recognition: 1600,
  merging: 600,
};

export function stagesForMedia(mediaKind: MediaKind): PipelineStage[] {
  const ids: PipelineStageId[] =
    mediaKind === "video"
      ? ["frame-sampling", "detection", "recognition", "merging"]
      : ["detection", "recognition", "merging"];

  return ids.map((id) => ({
    id,
    label: STAGE_LABELS[id],
    status: "pending",
    progress: 0,
  }));
}

export interface PipelineTickHandlers {
  onStageStart: (stageId: PipelineStageId) => void;
  onStageProgress: (stageId: PipelineStageId, progress: number) => void;
  onStageDone: (stageId: PipelineStageId) => void;
}

const FRAME_INTERVAL_MS = 50;

/** Animates each stage's progress bar with a simulated cadence; cancels cleanly via AbortSignal. */
export async function runPipelineSimulation(
  mediaKind: MediaKind,
  handlers: PipelineTickHandlers,
  signal: AbortSignal,
): Promise<void> {
  const stages = stagesForMedia(mediaKind);

  for (const stage of stages) {
    if (signal.aborted) return;
    handlers.onStageStart(stage.id);

    const duration = STAGE_DURATIONS_MS[stage.id];
    const steps = Math.max(1, Math.round(duration / FRAME_INTERVAL_MS));

    for (let step = 1; step <= steps; step += 1) {
      if (signal.aborted) return;
      await wait(FRAME_INTERVAL_MS);
      if (signal.aborted) return;
      handlers.onStageProgress(stage.id, Math.round((step / steps) * 100));
    }

    handlers.onStageDone(stage.id);
  }
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

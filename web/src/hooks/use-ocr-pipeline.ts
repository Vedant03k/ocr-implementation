"use client";

import { useCallback, useEffect, useReducer, useRef } from "react";

import { IMAGE_DETECTIONS, VIDEO_DETECTIONS, type RawDetection } from "@/lib/mock-fixtures";
import { runPipelineSimulation, stagesForMedia } from "@/lib/pipeline-stages";
import type { AppPhase, MediaKind, PipelineStage, PipelineStageId } from "@/types/ocr";

export const DEFAULT_CONFIDENCE_THRESHOLD = 0.6;

export interface UploadedMedia {
  file: File;
  previewUrl: string;
  mediaKind: MediaKind;
}

interface PipelineState {
  phase: AppPhase;
  media: UploadedMedia | null;
  stages: PipelineStage[];
  confidenceThreshold: number;
  rawDetections: RawDetection[] | null;
  videoDurationSeconds: number | null;
}

type Action =
  | { type: "MEDIA_SELECTED"; media: UploadedMedia }
  | { type: "RESET" }
  | { type: "SET_THRESHOLD"; value: number }
  | { type: "SET_VIDEO_DURATION"; seconds: number }
  | { type: "STAGE_START"; stageId: PipelineStageId }
  | { type: "STAGE_PROGRESS"; stageId: PipelineStageId; progress: number }
  | { type: "STAGE_DONE"; stageId: PipelineStageId }
  | { type: "PROCESSING_COMPLETE" };

const initialState: PipelineState = {
  phase: "upload",
  media: null,
  stages: [],
  confidenceThreshold: DEFAULT_CONFIDENCE_THRESHOLD,
  rawDetections: null,
  videoDurationSeconds: null,
};

function reducer(state: PipelineState, action: Action): PipelineState {
  switch (action.type) {
    case "MEDIA_SELECTED":
      return {
        ...initialState,
        confidenceThreshold: state.confidenceThreshold,
        media: action.media,
        phase: "processing",
        stages: stagesForMedia(action.media.mediaKind),
      };
    case "RESET":
      if (state.media) URL.revokeObjectURL(state.media.previewUrl);
      return { ...initialState, confidenceThreshold: state.confidenceThreshold };
    case "SET_THRESHOLD":
      return { ...state, confidenceThreshold: action.value };
    case "SET_VIDEO_DURATION":
      return { ...state, videoDurationSeconds: action.seconds };
    case "STAGE_START":
      return {
        ...state,
        stages: state.stages.map((stage) =>
          stage.id === action.stageId ? { ...stage, status: "running" } : stage,
        ),
      };
    case "STAGE_PROGRESS":
      return {
        ...state,
        stages: state.stages.map((stage) =>
          stage.id === action.stageId ? { ...stage, progress: action.progress } : stage,
        ),
      };
    case "STAGE_DONE":
      return {
        ...state,
        stages: state.stages.map((stage) =>
          stage.id === action.stageId ? { ...stage, status: "done", progress: 100 } : stage,
        ),
      };
    case "PROCESSING_COMPLETE":
      return {
        ...state,
        phase: "results",
        rawDetections: state.media?.mediaKind === "video" ? VIDEO_DETECTIONS : IMAGE_DETECTIONS,
      };
    default:
      return state;
  }
}

export function useOcrPipeline() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (state.phase !== "processing" || !state.media) return undefined;

    const controller = new AbortController();
    abortRef.current = controller;

    runPipelineSimulation(
      state.media.mediaKind,
      {
        onStageStart: (stageId) => dispatch({ type: "STAGE_START", stageId }),
        onStageProgress: (stageId, progress) =>
          dispatch({ type: "STAGE_PROGRESS", stageId, progress }),
        onStageDone: (stageId) => dispatch({ type: "STAGE_DONE", stageId }),
      },
      controller.signal,
    ).then(() => {
      if (!controller.signal.aborted) dispatch({ type: "PROCESSING_COMPLETE" });
    });

    return () => controller.abort();
  }, [state.phase, state.media]);

  const mediaRef = useRef(state.media);
  useEffect(() => {
    mediaRef.current = state.media;
  }, [state.media]);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      if (mediaRef.current) URL.revokeObjectURL(mediaRef.current.previewUrl);
    };
  }, []);

  const selectMedia = useCallback((file: File) => {
    const mediaKind: MediaKind = file.type.startsWith("video/") ? "video" : "image";
    const previewUrl = URL.createObjectURL(file);
    dispatch({ type: "MEDIA_SELECTED", media: { file, previewUrl, mediaKind } });
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    dispatch({ type: "RESET" });
  }, []);

  const setThreshold = useCallback((value: number) => {
    dispatch({ type: "SET_THRESHOLD", value });
  }, []);

  const setVideoDuration = useCallback((seconds: number) => {
    dispatch({ type: "SET_VIDEO_DURATION", seconds });
  }, []);

  return { state, selectMedia, reset, setThreshold, setVideoDuration };
}

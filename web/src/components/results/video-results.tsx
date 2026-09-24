"use client";

import { Search } from "lucide-react";
import { useMemo, useRef, useState } from "react";

import { ModelBadge } from "@/components/results/model-badge";
import { OcrBoxOverlay } from "@/components/results/ocr-box-overlay";
import type { OcrBox } from "@/types/ocr";

interface VideoResultsProps {
  previewUrl: string;
  boxes: OcrBox[];
  onDurationLoaded: (seconds: number) => void;
}

const FRAME_WINDOW_SECONDS = 0.9;

export function VideoResults({ previewUrl, boxes, onDurationLoaded }: VideoResultsProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);

  const boxesWithTimestamp = useMemo(
    () => boxes.filter((box): box is OcrBox & { timestamp: number } => box.timestamp !== undefined),
    [boxes],
  );

  const currentFrameBoxes = useMemo(
    () =>
      boxesWithTimestamp.filter(
        (box) => Math.abs(box.timestamp - currentTime) <= FRAME_WINDOW_SECONDS,
      ),
    [boxesWithTimestamp, currentTime],
  );

  const transcript = useMemo(() => {
    const sorted = [...boxesWithTimestamp].sort((a, b) => a.timestamp - b.timestamp);
    if (!query.trim()) return sorted;
    const needle = query.trim().toLowerCase();
    return sorted.filter((box) => box.text.toLowerCase().includes(needle));
  }, [boxesWithTimestamp, query]);

  const seekTo = (seconds: number) => {
    if (videoRef.current) videoRef.current.currentTime = seconds;
    setCurrentTime(seconds);
  };

  return (
    <div className="flex flex-col gap-4">
      <div>
        <div className="relative w-full overflow-hidden rounded-lg border border-zinc-200 bg-black dark:border-zinc-800">
          <video
            ref={videoRef}
            src={previewUrl}
            controls
            className="block w-full"
            onLoadedMetadata={(event) => onDurationLoaded(event.currentTarget.duration)}
            onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
          />
          <OcrBoxOverlay
            boxes={currentFrameBoxes}
            visible
            selectedId={selectedId}
            onSelect={(id) => setSelectedId((current) => (current === id ? null : id))}
            onHover={setHoveredId}
          />
        </div>
        <p className="mt-2 font-mono text-xs text-zinc-500 dark:text-zinc-400">
          {currentFrameBoxes.length} region{currentFrameBoxes.length === 1 ? "" : "s"} at{" "}
          {currentTime.toFixed(1)}s
        </p>
      </div>

      <div>
        <div className="relative">
          <Search className="pointer-events-none absolute top-1/2 left-2.5 h-3.5 w-3.5 -translate-y-1/2 text-zinc-400" />
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search transcript"
            className="w-full rounded-md border border-zinc-200 bg-transparent py-1.5 pr-3 pl-8 text-xs text-zinc-900 outline-none focus:border-zinc-400 dark:border-zinc-800 dark:text-zinc-100 dark:focus:border-zinc-600"
          />
        </div>

        <ul className="mt-2 flex max-h-80 flex-col divide-y divide-zinc-100 overflow-y-auto rounded-md border border-zinc-200 dark:divide-zinc-800 dark:border-zinc-800">
          {transcript.length === 0 ? (
            <li className="px-3 py-4 text-center text-xs text-zinc-400 dark:text-zinc-600">
              No matches
            </li>
          ) : (
            transcript.map((box) => (
              <li key={box.id}>
                <button
                  type="button"
                  onClick={() => {
                    seekTo(box.timestamp);
                    setSelectedId(box.id);
                  }}
                  onMouseEnter={() => setHoveredId(box.id)}
                  onMouseLeave={() => setHoveredId(null)}
                  className={`flex w-full items-start gap-3 px-3 py-2 text-left transition-colors hover:bg-zinc-50 dark:hover:bg-zinc-900 ${
                    (hoveredId ?? selectedId) === box.id ? "bg-zinc-50 dark:bg-zinc-900" : ""
                  }`}
                >
                  <span className="mt-0.5 shrink-0 font-mono text-xs text-zinc-400 dark:text-zinc-600">
                    {box.timestamp.toFixed(1)}s
                  </span>
                  <span className="flex-1 font-mono text-xs break-words text-zinc-900 dark:text-zinc-100">
                    {box.text}
                  </span>
                  <ModelBadge model={box.model} />
                </button>
              </li>
            ))
          )}
        </ul>
      </div>
    </div>
  );
}

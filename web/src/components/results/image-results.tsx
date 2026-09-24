"use client";

/* eslint-disable @next/next/no-img-element -- object URLs from user uploads aren't optimizable by next/image */

import { useMemo, useState } from "react";

import { BoxDetailPanel } from "@/components/results/box-detail-panel";
import { OcrBoxOverlay } from "@/components/results/ocr-box-overlay";
import type { OcrBox } from "@/types/ocr";

interface ImageResultsProps {
  previewUrl: string;
  boxes: OcrBox[];
}

export function ImageResults({ previewUrl, boxes }: ImageResultsProps) {
  const [boxesVisible, setBoxesVisible] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);

  const activeBox = useMemo(
    () => boxes.find((box) => box.id === (hoveredId ?? selectedId)) ?? null,
    [boxes, hoveredId, selectedId],
  );

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-zinc-500 dark:text-zinc-400">
          {boxes.length} text region{boxes.length === 1 ? "" : "s"} detected
        </span>
        <label className="flex items-center gap-2 text-xs text-zinc-600 dark:text-zinc-400">
          <input
            type="checkbox"
            checked={boxesVisible}
            onChange={(event) => setBoxesVisible(event.target.checked)}
            className="accent-zinc-900 dark:accent-zinc-100"
          />
          Show bounding boxes
        </label>
      </div>

      <div className="relative w-full overflow-hidden rounded-lg border border-zinc-200 dark:border-zinc-800">
        <img src={previewUrl} alt="Uploaded document" className="block w-full" />
        <OcrBoxOverlay
          boxes={boxes}
          visible={boxesVisible}
          selectedId={selectedId}
          onSelect={(id) => setSelectedId((current) => (current === id ? null : id))}
          onHover={setHoveredId}
        />
      </div>

      <BoxDetailPanel box={activeBox} />
    </div>
  );
}

"use client";

import type { OcrBox } from "@/types/ocr";

interface OcrBoxOverlayProps {
  boxes: OcrBox[];
  visible: boolean;
  selectedId: string | null;
  onSelect: (id: string) => void;
  onHover: (id: string | null) => void;
}

const MODEL_BORDER_CLASS: Record<OcrBox["model"], string> = {
  paddleocr: "border-emerald-500",
  "got-ocr2": "border-amber-500",
};

export function OcrBoxOverlay({
  boxes,
  visible,
  selectedId,
  onSelect,
  onHover,
}: OcrBoxOverlayProps) {
  if (!visible) return null;

  return (
    <div className="pointer-events-none absolute inset-0">
      {boxes.map((box) => {
        const isSelected = box.id === selectedId;
        return (
          <button
            key={box.id}
            type="button"
            onClick={() => onSelect(box.id)}
            onMouseEnter={() => onHover(box.id)}
            onMouseLeave={() => onHover(null)}
            onFocus={() => onHover(box.id)}
            onBlur={() => onHover(null)}
            className={`pointer-events-auto absolute border-[1.5px] transition-colors ${MODEL_BORDER_CLASS[box.model]} ${
              isSelected
                ? "bg-zinc-900/10 dark:bg-zinc-100/10"
                : "hover:bg-zinc-900/5 dark:hover:bg-zinc-100/5"
            }`}
            style={{
              left: `${box.x * 100}%`,
              top: `${box.y * 100}%`,
              width: `${box.w * 100}%`,
              height: `${box.h * 100}%`,
            }}
            title={`${box.text} (${Math.round(box.confidence * 100)}%)`}
            aria-label={`Box: ${box.text}`}
            aria-pressed={isSelected}
          />
        );
      })}
    </div>
  );
}

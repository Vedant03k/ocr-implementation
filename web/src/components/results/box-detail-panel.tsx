import { ModelBadge } from "@/components/results/model-badge";
import type { OcrBox } from "@/types/ocr";

export function BoxDetailPanel({ box }: { box: OcrBox | null }) {
  if (!box) {
    return (
      <div className="rounded-md border border-dashed border-zinc-200 px-3 py-4 text-center text-xs text-zinc-400 dark:border-zinc-800 dark:text-zinc-600">
        Hover or click a box to inspect its recognized text
      </div>
    );
  }

  return (
    <div className="rounded-md border border-zinc-200 px-3 py-3 dark:border-zinc-800">
      <p className="font-mono text-sm break-words text-zinc-900 dark:text-zinc-100">{box.text}</p>
      <div className="mt-2 flex items-center gap-2">
        <ModelBadge model={box.model} />
        <span className="font-mono text-xs text-zinc-500 dark:text-zinc-400">
          {Math.round(box.confidence * 100)}% confidence
        </span>
        {box.timestamp !== undefined ? (
          <span className="font-mono text-xs text-zinc-500 dark:text-zinc-400">
            @ {box.timestamp.toFixed(1)}s
          </span>
        ) : null}
      </div>
    </div>
  );
}

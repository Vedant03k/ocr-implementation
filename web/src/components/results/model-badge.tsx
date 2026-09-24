import type { OcrModel } from "@/types/ocr";

const MODEL_LABEL: Record<OcrModel, string> = {
  paddleocr: "PaddleOCR",
  "got-ocr2": "GOT-OCR2.0",
};

const MODEL_CLASSNAME: Record<OcrModel, string> = {
  paddleocr: "bg-emerald-50 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-400",
  "got-ocr2": "bg-amber-50 text-amber-700 dark:bg-amber-500/10 dark:text-amber-400",
};

export function ModelBadge({ model }: { model: OcrModel }) {
  return (
    <span
      className={`inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-medium ${MODEL_CLASSNAME[model]}`}
    >
      {MODEL_LABEL[model]}
    </span>
  );
}

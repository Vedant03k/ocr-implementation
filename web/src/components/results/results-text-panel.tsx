"use client";

import { Check, ChevronDown, Copy, FileJson, FileText } from "lucide-react";
import { useState } from "react";

import { resultToJson, resultToPlainText } from "@/lib/export-results";
import type { OcrResult } from "@/types/ocr";

type ViewMode = "text" | "json";

export function ResultsTextPanel({ result }: { result: OcrResult }) {
  const [isOpen, setIsOpen] = useState(true);
  const [mode, setMode] = useState<ViewMode>("text");
  const [copied, setCopied] = useState(false);

  const content = mode === "text" ? resultToPlainText(result) : resultToJson(result);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard API can be unavailable (e.g. insecure context); nothing to fall back to.
    }
  };

  return (
    <div className="rounded-lg border border-zinc-200 dark:border-zinc-800">
      <button
        type="button"
        onClick={() => setIsOpen((open) => !open)}
        className="flex w-full items-center justify-between px-4 py-3 text-left"
      >
        <span className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
          Recognized text
        </span>
        <ChevronDown
          className={`h-4 w-4 text-zinc-400 transition-transform ${isOpen ? "rotate-180" : ""}`}
        />
      </button>

      {isOpen ? (
        <div className="border-t border-zinc-200 dark:border-zinc-800">
          <div className="flex items-center justify-between px-4 pt-3">
            <div className="flex gap-1">
              <ModeTab
                label="Plain text"
                icon={FileText}
                active={mode === "text"}
                onClick={() => setMode("text")}
              />
              <ModeTab
                label="JSON"
                icon={FileJson}
                active={mode === "json"}
                onClick={() => setMode("json")}
              />
            </div>
            <button
              type="button"
              onClick={handleCopy}
              className="inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-xs font-medium text-zinc-500 transition-colors hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100"
            >
              {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
              {copied ? "Copied" : "Copy"}
            </button>
          </div>

          <pre className="m-4 mt-2 max-h-80 overflow-auto rounded-md bg-zinc-50 p-3 font-mono text-xs whitespace-pre-wrap text-zinc-700 dark:bg-zinc-900 dark:text-zinc-300">
            {content || "No text regions detected."}
          </pre>
        </div>
      ) : null}
    </div>
  );
}

function ModeTab({
  label,
  icon: Icon,
  active,
  onClick,
}: {
  label: string;
  icon: typeof FileText;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-xs font-medium transition-colors ${
        active
          ? "bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900"
          : "text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100"
      }`}
    >
      <Icon className="h-3.5 w-3.5" />
      {label}
    </button>
  );
}

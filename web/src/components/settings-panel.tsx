"use client";

import { ChevronDown, SlidersHorizontal } from "lucide-react";
import { useState } from "react";

import type { RoutingStats } from "@/lib/ocr-routing";

interface SettingsPanelProps {
  confidenceThreshold: number;
  onThresholdChange: (value: number) => void;
  routingStats: RoutingStats;
}

export function SettingsPanel({
  confidenceThreshold,
  onThresholdChange,
  routingStats,
}: SettingsPanelProps) {
  const [isOpen, setIsOpen] = useState(true);

  return (
    <div className="rounded-lg border border-zinc-200 dark:border-zinc-800">
      <button
        type="button"
        onClick={() => setIsOpen((open) => !open)}
        className="flex w-full items-center justify-between px-4 py-3 text-left"
      >
        <span className="flex items-center gap-2 text-sm font-medium text-zinc-900 dark:text-zinc-100">
          <SlidersHorizontal className="h-4 w-4" />
          Settings
        </span>
        <ChevronDown
          className={`h-4 w-4 text-zinc-400 transition-transform ${isOpen ? "rotate-180" : ""}`}
        />
      </button>

      {isOpen ? (
        <div className="border-t border-zinc-200 px-4 py-4 dark:border-zinc-800">
          <div className="flex items-center justify-between">
            <label
              htmlFor="confidence-threshold"
              className="text-xs font-medium text-zinc-700 dark:text-zinc-300"
            >
              Escalation threshold
            </label>
            <span className="font-mono text-xs text-zinc-500 dark:text-zinc-400">
              {confidenceThreshold.toFixed(2)}
            </span>
          </div>
          <input
            id="confidence-threshold"
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={confidenceThreshold}
            onChange={(event) => onThresholdChange(Number(event.target.value))}
            className="mt-2 w-full accent-zinc-900 dark:accent-zinc-100"
          />
          <p className="mt-1 text-[11px] leading-relaxed text-zinc-500 dark:text-zinc-500">
            Crops where PaddleOCR&apos;s confidence falls below this value escalate to GOT-OCR2.0.
          </p>

          <div className="mt-4 grid grid-cols-2 gap-3">
            <RoutingStat
              label="Fast path"
              sublabel="PaddleOCR"
              value={routingStats.fastPath}
              total={routingStats.total}
              barClassName="bg-emerald-500"
            />
            <RoutingStat
              label="Specialist"
              sublabel="GOT-OCR2.0"
              value={routingStats.specialist}
              total={routingStats.total}
              barClassName="bg-amber-500"
            />
          </div>
        </div>
      ) : null}
    </div>
  );
}

interface RoutingStatProps {
  label: string;
  sublabel: string;
  value: number;
  total: number;
  barClassName: string;
}

function RoutingStat({ label, sublabel, value, total, barClassName }: RoutingStatProps) {
  const percentage = total === 0 ? 0 : Math.round((value / total) * 100);

  return (
    <div className="rounded-md border border-zinc-200 p-2.5 dark:border-zinc-800">
      <div className="flex items-baseline justify-between">
        <span className="text-xs font-medium text-zinc-900 dark:text-zinc-100">{label}</span>
        <span className="font-mono text-xs text-zinc-500 dark:text-zinc-400">{value}</span>
      </div>
      <p className="text-[11px] text-zinc-500 dark:text-zinc-500">{sublabel}</p>
      <div className="mt-1.5 h-1 w-full overflow-hidden rounded-full bg-zinc-100 dark:bg-zinc-800">
        <div className={`h-full ${barClassName}`} style={{ width: `${percentage}%` }} />
      </div>
    </div>
  );
}

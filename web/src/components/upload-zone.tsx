"use client";

import { FileVideo, Image as ImageIcon, UploadCloud } from "lucide-react";
import { useCallback, useRef, useState } from "react";

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "video/mp4", "video/quicktime"];
const ACCEPT_ATTR = ".jpg,.jpeg,.png,.mp4,.mov,image/jpeg,image/png,video/mp4,video/quicktime";

interface UploadZoneProps {
  onFileSelected: (file: File) => void;
}

export function UploadZone({ onFileSelected }: UploadZoneProps) {
  const [isDragActive, setIsDragActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File | undefined) => {
      if (!file) return;
      if (!ACCEPTED_TYPES.includes(file.type)) {
        setError("Unsupported file type. Upload a JPG, PNG, MP4, or MOV.");
        return;
      }
      setError(null);
      onFileSelected(file);
    },
    [onFileSelected],
  );

  return (
    <div className="w-full">
      <div
        role="button"
        tabIndex={0}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") inputRef.current?.click();
        }}
        onDragOver={(event) => {
          event.preventDefault();
          setIsDragActive(true);
        }}
        onDragLeave={() => setIsDragActive(false)}
        onDrop={(event) => {
          event.preventDefault();
          setIsDragActive(false);
          handleFile(event.dataTransfer.files[0]);
        }}
        className={`flex cursor-pointer flex-col items-center justify-center gap-3 rounded-lg border border-dashed px-6 py-16 text-center transition-colors ${
          isDragActive
            ? "border-zinc-900 bg-zinc-50 dark:border-zinc-100 dark:bg-zinc-900"
            : "border-zinc-300 hover:border-zinc-400 dark:border-zinc-700 dark:hover:border-zinc-600"
        }`}
      >
        <UploadCloud className="h-8 w-8 text-zinc-400 dark:text-zinc-500" />
        <div>
          <p className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
            Drag and drop a file, or click to browse
          </p>
          <p className="mt-1 flex items-center justify-center gap-3 text-xs text-zinc-500 dark:text-zinc-500">
            <span className="inline-flex items-center gap-1">
              <ImageIcon className="h-3.5 w-3.5" /> JPG, PNG
            </span>
            <span className="inline-flex items-center gap-1">
              <FileVideo className="h-3.5 w-3.5" /> MP4, MOV
            </span>
          </p>
        </div>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT_ATTR}
          className="hidden"
          onChange={(event) => handleFile(event.target.files?.[0])}
        />
      </div>
      {error ? <p className="mt-2 text-xs text-red-600 dark:text-red-400">{error}</p> : null}
    </div>
  );
}

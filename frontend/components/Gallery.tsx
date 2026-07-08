"use client";

import { useCallback, useEffect, useState } from "react";
import { thumbSrc } from "@/lib/api";
import { Frame } from "@/lib/types";

export default function Gallery({
  frames,
  columns = 4,
  maxHeight = "26rem",
  emptyText = "No frames yet.",
}: {
  frames: Frame[];
  columns?: number;
  maxHeight?: string;
  emptyText?: string;
}) {
  const [index, setIndex] = useState<number | null>(null);

  if (!frames.length) {
    return <p className="text-sm text-neutral-500">{emptyText}</p>;
  }

  return (
    <>
      <div
        className="custom-scroll grid gap-3 overflow-y-auto pr-1"
        style={{
          gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
          maxHeight,
        }}
      >
        {frames.map((f, i) => (
          <figure
            key={`${f.video_id}-${f.frame_idx}`}
            onClick={() => setIndex(i)}
            className="group relative cursor-zoom-in overflow-hidden rounded-lg border border-neutral-800 bg-neutral-900 transition hover:border-emerald-600"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={thumbSrc(f.thumbnail_b64)}
              alt={f.caption}
              className="h-32 w-full object-cover transition group-hover:opacity-90"
            />
            <figcaption className="absolute inset-x-0 bottom-0 bg-black/75 px-2 py-1 text-[11px] leading-tight backdrop-blur-sm">
              <span className="font-mono text-emerald-400">
                t={f.timestamp_sec.toFixed(1)}s
              </span>
              {f.score > 0 && (
                <span className="text-neutral-400"> · {f.score.toFixed(2)}</span>
              )}
              <span className="block truncate text-neutral-300" title={f.caption}>
                {f.caption || "—"}
              </span>
            </figcaption>
          </figure>
        ))}
      </div>

      {index !== null && (
        <Lightbox
          frames={frames}
          index={index}
          onIndexChange={setIndex}
          onClose={() => setIndex(null)}
        />
      )}
    </>
  );
}

function Lightbox({
  frames,
  index,
  onIndexChange,
  onClose,
}: {
  frames: Frame[];
  index: number;
  onIndexChange: (i: number) => void;
  onClose: () => void;
}) {
  const frame = frames[index];
  const total = frames.length;

  const go = useCallback(
    (delta: number) => onIndexChange((index + delta + total) % total),
    [index, total, onIndexChange],
  );

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
      else if (e.key === "ArrowLeft") go(-1);
      else if (e.key === "ArrowRight") go(1);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go, onClose]);

  return (
    <div
      onClick={onClose}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4 backdrop-blur-sm"
    >
      {/* Prev / Next arrows (outside the card so they never scroll away). */}
      {total > 1 && (
        <>
          <ArrowButton
            side="left"
            onClick={(e) => {
              e.stopPropagation();
              go(-1);
            }}
          />
          <ArrowButton
            side="right"
            onClick={(e) => {
              e.stopPropagation();
              go(1);
            }}
          />
        </>
      )}

      <div
        onClick={(e) => e.stopPropagation()}
        className="custom-scroll relative flex flex-col overflow-auto rounded-2xl border border-neutral-700 bg-neutral-900 shadow-2xl"
        style={{
          // Drag the bottom-right corner to resize.
          resize: "both",
          width: "min(48rem, 92vw)",
          height: "min(85vh, 48rem)",
          minWidth: "20rem",
          minHeight: "18rem",
          maxWidth: "95vw",
          maxHeight: "92vh",
        }}
      >
        <div className="sticky top-0 z-10 flex items-center justify-between border-b border-neutral-800 bg-neutral-900/95 px-4 py-3 backdrop-blur">
          <span className="font-mono text-sm text-emerald-400">
            t={frame.timestamp_sec.toFixed(2)}s
            {frame.score > 0 && (
              <span className="ml-2 text-neutral-400">
                match {frame.score.toFixed(3)}
              </span>
            )}
          </span>
          <div className="flex items-center gap-3">
            <span className="text-xs text-neutral-500">
              {index + 1} / {total}
            </span>
            <button
              onClick={onClose}
              className="rounded-lg px-2 py-1 text-neutral-400 transition hover:bg-neutral-800 hover:text-white"
              title="Close (Esc)"
            >
              ✕
            </button>
          </div>
        </div>

        <div className="flex flex-1 items-center justify-center bg-black p-2">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={thumbSrc(frame.thumbnail_b64)}
            alt={frame.caption}
            className="max-h-full w-auto max-w-full rounded-lg object-contain"
          />
        </div>

        <div className="space-y-3 px-4 py-4">
          <div>
            <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-neutral-500">
              Caption
            </h4>
            <p className="text-sm leading-relaxed text-neutral-100">
              {frame.caption || "—"}
            </p>
          </div>

          <div>
            <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-neutral-500">
              Detected objects
            </h4>
            {frame.detection_labels.length ? (
              <div className="flex flex-wrap gap-1.5">
                {frame.detection_labels.map((label) => (
                  <span
                    key={label}
                    className="rounded-full border border-neutral-700 bg-neutral-800 px-2.5 py-0.5 text-xs text-neutral-200"
                  >
                    {label}
                  </span>
                ))}
              </div>
            ) : (
              <p className="text-sm text-neutral-500">None</p>
            )}
          </div>

          <div className="flex gap-6 pt-1 text-xs text-neutral-500">
            <span>
              frame&nbsp;
              <span className="font-mono text-neutral-300">
                #{frame.frame_idx}
              </span>
            </span>
            <span>
              video&nbsp;
              <span className="font-mono text-neutral-300">
                {frame.video_id}
              </span>
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

function ArrowButton({
  side,
  onClick,
}: {
  side: "left" | "right";
  onClick: (e: React.MouseEvent) => void;
}) {
  return (
    <button
      onClick={onClick}
      aria-label={side === "left" ? "Previous frame" : "Next frame"}
      className={`absolute top-1/2 z-10 flex h-11 w-11 -translate-y-1/2 items-center justify-center rounded-full border border-neutral-700 bg-neutral-900/80 text-neutral-200 transition hover:border-emerald-600 hover:bg-neutral-800 hover:text-white ${
        side === "left" ? "left-2 sm:left-6" : "right-2 sm:right-6"
      }`}
    >
      <svg
        width="20"
        height="20"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        {side === "left" ? (
          <path d="M15 18l-6-6 6-6" />
        ) : (
          <path d="M9 18l6-6-6-6" />
        )}
      </svg>
    </button>
  );
}

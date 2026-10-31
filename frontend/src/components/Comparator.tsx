import { useCallback, useEffect, useRef, useState } from "react";
import type { Hotspot } from "../types";

interface ComparatorProps {
  baseSrc: string;
  overlaySrc: string;
  hotspot?: Hotspot;
  /** Changing this re-runs the reveal animation, so pass something per-result. */
  revealKey: string;
}

const prefersReducedMotion = () =>
  typeof window !== "undefined" &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/**
 * Drag the divider to wipe between the photograph and its heatmap.
 *
 * Showing the two images side by side makes the eye travel; wiping them in place keeps
 * every pixel where it was, so a highlighted region can be matched to the skin underneath
 * it without moving your gaze.
 */
export default function Comparator({
  baseSrc,
  overlaySrc,
  hotspot,
  revealKey,
}: ComparatorProps) {
  const [split, setSplit] = useState(100);
  const frameRef = useRef<HTMLDivElement>(null);
  const draggingRef = useRef(false);

  // One orchestrated moment per result: the heatmap wipes in from the right, then stops
  // at halfway and waits for the user. Nothing else on the page animates on arrival.
  useEffect(() => {
    if (prefersReducedMotion()) {
      setSplit(50);
      return;
    }
    setSplit(100);
    let raf = 0;
    const started = performance.now();
    const duration = 620;

    const step = (now: number) => {
      const t = Math.min(1, (now - started) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      setSplit(100 - eased * 50);
      if (t < 1) {
        raf = requestAnimationFrame(step);
      }
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [revealKey]);

  const moveTo = useCallback((clientX: number) => {
    const frame = frameRef.current;
    if (!frame) return;
    const bounds = frame.getBoundingClientRect();
    const ratio = (clientX - bounds.left) / bounds.width;
    setSplit(Math.min(100, Math.max(0, ratio * 100)));
  }, []);

  useEffect(() => {
    const onMove = (event: PointerEvent) => {
      if (draggingRef.current) moveTo(event.clientX);
    };
    const onUp = () => {
      draggingRef.current = false;
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, [moveTo]);

  return (
    <div
      ref={frameRef}
      className="comparator"
      style={{ ["--split" as string]: `${split}%` }}
      onPointerDown={(event) => {
        draggingRef.current = true;
        moveTo(event.clientX);
      }}
      role="slider"
      tabIndex={0}
      aria-label="Wipe between the photograph and the heatmap"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(split)}
      aria-valuetext={`Heatmap covers ${Math.round(100 - split)} percent`}
      onKeyDown={(event) => {
        if (event.key === "ArrowLeft") {
          event.preventDefault();
          setSplit((value) => Math.max(0, value - 4));
        }
        if (event.key === "ArrowRight") {
          event.preventDefault();
          setSplit((value) => Math.min(100, value + 4));
        }
      }}
    >
      <img src={overlaySrc} alt="Heatmap of the regions that drove the prediction" />
      <img
        className="comparator__overlay"
        src={baseSrc}
        alt="The photograph as the model received it"
      />

      {hotspot && (
        <div
          className="comparator__box"
          style={{
            left: `${hotspot.bbox.x * 100}%`,
            top: `${hotspot.bbox.y * 100}%`,
            width: `${hotspot.bbox.width * 100}%`,
            height: `${hotspot.bbox.height * 100}%`,
          }}
        >
          <span>{Math.round(hotspot.evidence_share * 100)}% of evidence</span>
        </div>
      )}

      <div className="comparator__handle">
        <span className="comparator__grip" aria-hidden="true">
          &#8942;
        </span>
      </div>
      <span className="comparator__tag comparator__tag--left">heatmap</span>
      <span className="comparator__tag comparator__tag--right">photograph</span>
    </div>
  );
}

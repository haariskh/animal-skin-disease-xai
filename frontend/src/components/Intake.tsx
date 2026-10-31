import { useRef, useState } from "react";

interface IntakeProps {
  onSelect: (file: File) => void;
  maxUploadMb: number;
  disabled: boolean;
}

const ACCEPTED = ["image/jpeg", "image/png", "image/webp", "image/bmp"];

export default function Intake({ onSelect, maxUploadMb, disabled }: IntakeProps) {
  const [dragging, setDragging] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  /** Check type and size here so an obvious mistake never costs a round trip. */
  function accept(file: File | undefined) {
    if (!file) {
      return;
    }
    if (!ACCEPTED.includes(file.type)) {
      setProblem(`${file.name} is a ${file.type || "unknown"} file. Use a JPEG, PNG, WebP or BMP.`);
      return;
    }
    if (file.size > maxUploadMb * 1024 * 1024) {
      setProblem(
        `${file.name} is ${(file.size / 1024 / 1024).toFixed(1)} MB. The limit is ${maxUploadMb} MB.`,
      );
      return;
    }
    setProblem(null);
    onSelect(file);
  }

  return (
    <section className="intake">
      <p className="intake__lede">
        Drop in a close photograph of the affected skin. Fill the frame with the area in
        question and keep it in focus &mdash; the model only sees what you give it.
      </p>

      <div
        className={`dropzone${dragging ? " dropzone--over" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          if (!disabled) accept(event.dataTransfer.files?.[0]);
        }}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            inputRef.current?.click();
          }
        }}
        role="button"
        tabIndex={0}
        aria-label="Choose an image to analyse"
      >
        <span className="dropzone__headline">Drop an image here</span>
        <p className="dropzone__hint">
          or click to browse &middot; JPEG, PNG, WebP or BMP up to {maxUploadMb} MB
        </p>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED.join(",")}
          disabled={disabled}
          onChange={(event) => {
            accept(event.target.files?.[0]);
            // Clearing the value lets the same file be chosen twice in a row.
            event.target.value = "";
          }}
        />
      </div>

      {problem && (
        <div className="notice notice--error" style={{ marginTop: "1rem" }} role="alert">
          <p>{problem}</p>
        </div>
      )}

      <div className="primer">
        <h2 className="primer__title">What comes back</h2>
        <dl className="primer__list">
          <div>
            <dt>A reading</dt>
            <dd>
              The most likely condition with its probability, and how close the runner-up
              came. A narrow margin is reported rather than hidden.
            </dd>
          </div>
          <div>
            <dt>Where it looked</dt>
            <dd>
              A heatmap you can wipe against the original, plus how much of the evidence
              sits in each part of the frame.
            </dd>
          </div>
          <div>
            <dt>Whether to believe it</dt>
            <dd>
              The highlighted pixels are removed and re-added to measure whether they
              really drove the result, scored against a random ranking.
            </dd>
          </div>
        </dl>
      </div>
    </section>
  );
}

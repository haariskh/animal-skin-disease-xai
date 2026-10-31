import type { Health, Metadata } from "../types";

interface RailProps {
  metadata: Metadata | null;
  health: Health | null;
}

function formatTrainedAt(iso: string): string {
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) {
    return "unknown";
  }
  return parsed.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

/** Headline metric from the checkpoint, when training recorded one. */
function headlineMetric(metrics: Record<string, unknown>): string | null {
  const monitor = typeof metrics.monitor === "string" ? metrics.monitor : null;
  const best = typeof metrics.best_metric === "number" ? metrics.best_metric : null;
  if (!monitor || best === null) {
    return null;
  }
  return `${monitor.replace("val_", "")} ${best.toFixed(3)}`;
}

export default function Rail({ metadata, health }: RailProps) {
  const online = health?.model_loaded ?? false;
  const metric = metadata ? headlineMetric(metadata.metrics) : null;

  return (
    <aside className="rail">
      <div className="masthead">
        <h1 className="masthead__name">Aniderm</h1>
        <p className="masthead__what">
          Reads an animal skin photograph, names the most likely condition, and shows
          which part of the picture led it there.
        </p>
      </div>

      {metadata && (
        <div className="block">
          <h2 className="block__title">Model</h2>
          <dl className="spec">
            <dt>Network</dt>
            <dd>{metadata.backbone}</dd>
            <dt>Input</dt>
            <dd>
              {metadata.image_size}&times;{metadata.image_size}
            </dd>
            <dt>Trained on</dt>
            <dd>{metadata.dataset}</dd>
            <dt>Trained</dt>
            <dd>{formatTrainedAt(metadata.trained_at)}</dd>
            <dt>Running on</dt>
            <dd>{metadata.device}</dd>
            {metric && (
              <>
                <dt>Validation</dt>
                <dd>{metric}</dd>
              </>
            )}
          </dl>
        </div>
      )}

      {metadata && (
        <div className="block">
          <h2 className="block__title">What it can recognise</h2>
          <ul className="legend">
            {metadata.classes.map((item) => (
              <li key={item.key}>
                <span className="legend__swatch" style={{ background: item.colour }} />
                <span className="legend__body">
                  <strong>{item.display_name}</strong>
                  <span>{item.summary}</span>
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="block">
        <span className="pulse">
          <span className={`pulse__dot${online ? "" : " pulse__dot--down"}`} />
          {online ? "Model loaded" : "No model loaded"}
        </span>
      </div>

      <p className="rail__foot">
        A research prototype, not a diagnostic device. Anything it reports needs a
        veterinarian before it means anything.
      </p>
    </aside>
  );
}

import type { Faithfulness } from "../types";

interface TrustProps {
  faithfulness: Faithfulness;
}

const WIDTH = 460;
const HEIGHT = 190;
const PAD = { top: 14, right: 14, bottom: 26, left: 34 };

function toPath(values: number[], fractions: number[]): string {
  const innerW = WIDTH - PAD.left - PAD.right;
  const innerH = HEIGHT - PAD.top - PAD.bottom;
  return values
    .map((value, index) => {
      const x = PAD.left + (fractions[index] ?? index / (values.length - 1)) * innerW;
      const y = PAD.top + (1 - Math.min(1, Math.max(0, value))) * innerH;
      return `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

/**
 * Deletion and insertion curves.
 *
 * The question this answers is not "where did the model look" but "is that heatmap
 * telling the truth". If removing the highlighted pixels barely moves the prediction,
 * the picture was decoration.
 */
export default function Trust({ faithfulness }: TrustProps) {
  if (faithfulness.error) {
    return <p className="reading">The faithfulness check failed: {faithfulness.error}</p>;
  }

  const fractions = faithfulness.fractions ?? [];
  const deletion = faithfulness.deletion_curve ?? [];
  const insertion = faithfulness.insertion_curve ?? [];
  if (!deletion.length || !insertion.length) {
    return <p className="reading">No faithfulness measurement was taken for this image.</p>;
  }

  const randomInsertion = faithfulness.random_insertion_auc;
  const innerH = HEIGHT - PAD.top - PAD.bottom;

  return (
    <div className="trust">
      <svg
        className="curves"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label="Deletion and insertion curves for the current explanation"
      >
        {[0, 0.5, 1].map((tick) => (
          <g key={tick}>
            <line
              x1={PAD.left}
              x2={WIDTH - PAD.right}
              y1={PAD.top + (1 - tick) * innerH}
              y2={PAD.top + (1 - tick) * innerH}
              stroke="var(--line-soft)"
              strokeWidth="1"
            />
            <text
              x={PAD.left - 7}
              y={PAD.top + (1 - tick) * innerH + 3.5}
              textAnchor="end"
              fill="var(--ink-faint)"
              fontSize="9"
              fontFamily="var(--mono)"
            >
              {tick.toFixed(1)}
            </text>
          </g>
        ))}

        <path d={toPath(insertion, fractions)} fill="none" stroke="var(--signal)" strokeWidth="2" />
        <path
          d={toPath(deletion, fractions)}
          fill="none"
          stroke="var(--alert)"
          strokeWidth="2"
          strokeDasharray="4 3"
        />

        <text x={PAD.left} y={HEIGHT - 8} fill="var(--ink-faint)" fontSize="9" fontFamily="var(--mono)">
          0% of pixels
        </text>
        <text
          x={WIDTH - PAD.right}
          y={HEIGHT - 8}
          textAnchor="end"
          fill="var(--ink-faint)"
          fontSize="9"
          fontFamily="var(--mono)"
        >
          100%
        </text>

        <g transform={`translate(${PAD.left + 8}, ${PAD.top + 10})`}>
          <line x1="0" x2="16" y1="0" y2="0" stroke="var(--signal)" strokeWidth="2" />
          <text x="22" y="3.5" fill="var(--ink-dim)" fontSize="10">
            inserting the highlighted pixels
          </text>
          <line
            x1="0"
            x2="16"
            y1="14"
            y2="14"
            stroke="var(--alert)"
            strokeWidth="2"
            strokeDasharray="4 3"
          />
          <text x="22" y="17.5" fill="var(--ink-dim)" fontSize="10">
            removing them
          </text>
        </g>
      </svg>

      <div>
        <dl className="trust__numbers">
          <dt>Insertion</dt>
          <dd>{faithfulness.insertion_auc?.toFixed(3) ?? "—"}</dd>
          <dt>Deletion</dt>
          <dd>{faithfulness.deletion_auc?.toFixed(3) ?? "—"}</dd>
          {randomInsertion !== undefined && (
            <>
              <dt>Random control</dt>
              <dd>{randomInsertion.toFixed(3)}</dd>
            </>
          )}
        </dl>
        <p className="trust__verdict">{faithfulness.verdict}</p>
      </div>
    </div>
  );
}

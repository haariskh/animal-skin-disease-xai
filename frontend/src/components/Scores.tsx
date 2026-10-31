import type { Score } from "../types";

interface ScoresProps {
  scores: Score[];
  explainedKey: string;
  onExplainOther: (key: string) => void;
  busy: boolean;
}

export default function Scores({
  scores,
  explainedKey,
  onExplainOther,
  busy,
}: ScoresProps) {
  return (
    <ul className="scores">
      {scores.map((score, index) => {
        const isExplained = score.key === explainedKey;
        return (
          <li key={score.key} className={index === 0 ? "score" : "score score--muted"}>
            <div className="score__row">
              <span className="score__name">{score.display_name}</span>
              <span className="score__value">{(score.probability * 100).toFixed(1)}%</span>
            </div>
            <div className="score__track">
              <div
                className="score__fill"
                style={{
                  width: `${Math.max(score.probability * 100, 0.6)}%`,
                  background: score.colour,
                  opacity: isExplained ? 1 : 0.55,
                }}
              />
            </div>
            {!isExplained && (
              <button
                type="button"
                className="score__explain"
                disabled={busy}
                onClick={() => onExplainOther(score.key)}
              >
                Show the evidence for {score.display_name.toLowerCase()} instead
              </button>
            )}
          </li>
        );
      })}
    </ul>
  );
}

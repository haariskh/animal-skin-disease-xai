import { useCallback, useEffect, useMemo, useState } from "react";
import { analyzeImage, fetchHealth, fetchMetadata } from "./api";
import Comparator from "./components/Comparator";
import Evidence from "./components/Evidence";
import Intake from "./components/Intake";
import MethodTabs from "./components/MethodTabs";
import Rail from "./components/Rail";
import Scores from "./components/Scores";
import Trust from "./components/Trust";
import type { AnalyzeResponse, Health, Metadata } from "./types";

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [activeMethod, setActiveMethod] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const status = await fetchHealth();
        if (cancelled) return;
        setHealth(status);
        if (status.model_loaded) {
          setMetadata(await fetchMetadata());
        }
      } catch (cause) {
        if (!cancelled) {
          setError(
            cause instanceof Error
              ? `Cannot reach the API: ${cause.message}`
              : "Cannot reach the API.",
          );
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const run = useCallback(
    async (chosen: File, targetClass?: string) => {
      setBusy(true);
      setError(null);
      try {
        const response = await analyzeImage({ file: chosen, targetClass });
        setResult(response);
        setActiveMethod((current) => {
          const stillThere = response.explanations.some((item) => item.method === current);
          return stillThere ? current : (response.explanations[0]?.method ?? "");
        });
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : "The analysis failed.");
      } finally {
        setBusy(false);
      }
    },
    [],
  );

  const onSelect = useCallback(
    (chosen: File) => {
      setFile(chosen);
      setResult(null);
      void run(chosen);
    },
    [run],
  );

  const explanation = useMemo(
    () => result?.explanations.find((item) => item.method === activeMethod),
    [result, activeMethod],
  );

  const modelMissing = health !== null && !health.model_loaded;

  return (
    <div className="shell">
      <a className="skip-link" href="#stage">
        Skip to the main panel
      </a>
      <Rail metadata={metadata} health={health} />

      <main className="stage" id="stage">
        {modelMissing && (
          <div className="notice" role="status">
            <p>No trained model is loaded, so nothing can be analysed yet.</p>
            <pre>{health?.detail}</pre>
          </div>
        )}

        {error && (
          <div className="notice notice--error" role="alert" style={{ marginBottom: "1.5rem" }}>
            <p>{error}</p>
          </div>
        )}

        {!result && !busy && (
          <Intake
            onSelect={onSelect}
            maxUploadMb={metadata?.max_upload_mb ?? 10}
            disabled={modelMissing}
          />
        )}

        {busy && (
          <div className="working" aria-live="polite">
            <span>Classifying and building the explanations</span>
            <span className="working__bar" />
          </div>
        )}

        {result && !busy && (
          <div className="results">
            <div className="results__head">
              <div>
                <div
                  className="verdict"
                  style={{ ["--verdict-colour" as string]: result.prediction.colour }}
                >
                  <span className="verdict__name">{result.prediction.display_name}</span>
                  <span className="verdict__value">
                    {(result.prediction.probability * 100).toFixed(1)}%
                  </span>
                </div>
                {result.prediction.summary && (
                  <p className="verdict__note">{result.prediction.summary}</p>
                )}
              </div>
              <button
                type="button"
                className="button button--quiet"
                onClick={() => {
                  setResult(null);
                  setFile(null);
                  setError(null);
                }}
              >
                Analyse another image
              </button>
            </div>

            {result.warnings.map((warning) => (
              <div className="notice" key={warning} role="status">
                <p>{warning}</p>
              </div>
            ))}

            <section className="section">
              <div className="section__head">
                <h2 className="section__title">
                  Where the evidence for {result.explained_class.display_name.toLowerCase()} is
                </h2>
                {explanation && (
                  <span className="section__aside">{explanation.runtime_ms.toFixed(0)} ms</span>
                )}
              </div>

              <div className="viewer">
                {explanation && (
                  <Comparator
                    baseSrc={result.images.model_input}
                    overlaySrc={explanation.overlay}
                    hotspot={explanation.regions.hotspots[0]}
                    revealKey={`${file?.name ?? ""}-${explanation.method}-${result.explained_class.key}`}
                  />
                )}
                <div>
                  <MethodTabs
                    explanations={result.explanations}
                    active={activeMethod}
                    onChange={setActiveMethod}
                  />
                </div>
              </div>

              {explanation && (
                <Evidence
                  regions={explanation.regions}
                  accent={result.prediction.colour}
                />
              )}
            </section>

            <section className="section">
              <div className="section__head">
                <h2 className="section__title">What else it considered</h2>
                <span className="section__aside">
                  margin {(result.prediction.margin * 100).toFixed(1)}%
                </span>
              </div>
              <Scores
                scores={result.scores}
                explainedKey={result.explained_class.key}
                busy={busy}
                onExplainOther={(key) => {
                  if (file) void run(file, key);
                }}
              />
            </section>

            {explanation && Object.keys(explanation.faithfulness).length > 0 && (
              <section className="section">
                <div className="section__head">
                  <h2 className="section__title">Can this heatmap be trusted?</h2>
                  <span className="section__aside">{explanation.method_label}</span>
                </div>
                <Trust faithfulness={explanation.faithfulness} />
              </section>
            )}

            <p className="disclaimer">
              This is a research prototype trained on a small public dataset. It cannot
              diagnose an animal, and a confident-looking percentage is not a second
              opinion. Take anything it finds to a veterinarian.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}

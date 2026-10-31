import type { Regions } from "../types";

interface EvidenceProps {
  regions: Regions;
  accent: string;
}

/** Fade the accent colour by share, so the 3x3 grid reads as data, not decoration. */
function cellStyle(share: number, peak: number, accent: string) {
  const intensity = peak > 0 ? share / peak : 0;
  return {
    background: `color-mix(in srgb, ${accent} ${Math.round(intensity * 70)}%, var(--surface))`,
  };
}

export default function Evidence({ regions, accent }: EvidenceProps) {
  const peak = Math.max(...regions.grid.map((cell) => cell.share), 0.0001);
  const ordered = [...regions.grid].sort((a, b) => a.row - b.row || a.col - b.col);

  return (
    <div className="evidence">
      <div
        className="heatgrid"
        role="img"
        aria-label="Share of the evidence in each ninth of the image"
      >
        {ordered.map((cell) => (
          <div
            key={`${cell.row}-${cell.col}`}
            className="heatgrid__cell"
            style={cellStyle(cell.share, peak, accent)}
            title={`${cell.name}: ${(cell.share * 100).toFixed(0)}%`}
          >
            {(cell.share * 100).toFixed(0)}
          </div>
        ))}
      </div>

      <div>
        <p className="reading">{regions.summary}</p>

        {regions.hotspots.length > 0 ? (
          <ul className="hotspots">
            {regions.hotspots.map((spot, index) => (
              <li className="hotspot" key={`${spot.centroid.x}-${spot.centroid.y}-${index}`}>
                <span className="hotspot__rank">{index + 1}</span>
                <span>
                  Towards the {spot.location}
                  <br />
                  <span className="hotspot__meta">
                    {(spot.evidence_share * 100).toFixed(0)}% of evidence &middot; covers{" "}
                    {(spot.area_fraction * 100).toFixed(1)}% of the frame &middot; peak{" "}
                    {spot.peak_intensity.toFixed(2)}
                  </span>
                </span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="reading">
            No single area passed the intensity threshold, so the model drew on the image
            as a whole rather than one lesion.
          </p>
        )}
      </div>
    </div>
  );
}

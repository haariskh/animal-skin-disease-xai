/** Mirrors the pydantic models in api/schemas.py. */

export interface ClassInfo {
  key: string;
  display_name: string;
  summary: string;
  colour: string;
}

export interface MethodInfo {
  name: string;
  label: string;
  description: string;
  is_default: boolean;
}

export interface Metadata {
  backbone: string;
  image_size: number;
  dataset: string;
  trained_at: string;
  num_classes: number;
  device: string;
  classes: ClassInfo[];
  methods: MethodInfo[];
  metrics: Record<string, unknown>;
  low_confidence_threshold: number;
  max_upload_mb: number;
}

export interface Score {
  key: string;
  display_name: string;
  probability: number;
  colour: string;
  summary: string;
}

export interface Prediction {
  class_key: string;
  display_name: string;
  probability: number;
  colour: string;
  summary: string;
  margin: number;
  entropy: number;
}

export interface GridCell {
  name: string;
  row: number;
  col: number;
  share: number;
  mean_intensity: number;
}

export interface Hotspot {
  area_fraction: number;
  evidence_share: number;
  peak_intensity: number;
  mean_intensity: number;
  bbox: { x: number; y: number; width: number; height: number };
  centroid: { x: number; y: number };
  location: string;
}

export interface Regions {
  grid: GridCell[];
  hotspots: Hotspot[];
  concentration: {
    top_10pct_share: number;
    coverage_above_half: number;
    gini: number;
  };
  threshold: number;
  summary: string;
}

export interface Faithfulness {
  fractions?: number[];
  deletion_curve?: number[];
  insertion_curve?: number[];
  deletion_auc?: number;
  insertion_auc?: number;
  random_deletion_auc?: number;
  random_insertion_auc?: number;
  net_score?: number;
  verdict?: string;
  error?: string;
}

export interface Explanation {
  method: string;
  method_label: string;
  description: string;
  target_class_key: string;
  runtime_ms: number;
  parameters: Record<string, unknown>;
  regions: Regions;
  faithfulness: Faithfulness;
  overlay: string;
  heatmap: string;
}

export interface AnalyzeResponse {
  prediction: Prediction;
  scores: Score[];
  explained_class: {
    index: number;
    key: string;
    display_name: string;
    is_top_prediction: boolean;
  };
  explanations: Explanation[];
  images: { source: string; model_input: string };
  warnings: string[];
  timings_ms: Record<string, number>;
}

export interface Health {
  status: string;
  model_loaded: boolean;
  detail: string;
}

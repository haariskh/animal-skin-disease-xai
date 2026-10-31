"""Pydantic models for the API surface.

Declaring the responses explicitly means FastAPI generates accurate OpenAPI docs at
``/docs``, and the frontend's TypeScript types can be written straight off them.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ClassInfo(BaseModel):
    key: str
    display_name: str
    summary: str = ""
    colour: str = "#6b7280"


class MethodInfo(BaseModel):
    name: str
    label: str
    description: str
    is_default: bool = False


class HealthResponse(BaseModel):
    status: str = Field(description="'ok' when a model is loaded, 'degraded' otherwise")
    model_loaded: bool
    detail: str = ""


class MetadataResponse(BaseModel):
    backbone: str
    image_size: int
    dataset: str
    trained_at: str
    num_classes: int
    device: str
    classes: list[ClassInfo]
    methods: list[MethodInfo]
    metrics: dict[str, Any] = Field(default_factory=dict)
    low_confidence_threshold: float
    max_upload_mb: int


class ScoreOut(BaseModel):
    key: str
    display_name: str
    probability: float
    colour: str
    summary: str = ""


class PredictionOut(BaseModel):
    class_key: str
    display_name: str
    probability: float
    colour: str
    summary: str = ""
    margin: float
    entropy: float


class ExplainedClassOut(BaseModel):
    index: int
    key: str
    display_name: str
    is_top_prediction: bool


class ExplanationOut(BaseModel):
    method: str
    method_label: str
    description: str = ""
    target_class_key: str
    runtime_ms: float
    parameters: dict[str, Any] = Field(default_factory=dict)
    regions: dict[str, Any] = Field(default_factory=dict)
    faithfulness: dict[str, Any] = Field(default_factory=dict)
    overlay: str = Field(description="PNG data URI of the heatmap blended over the image")
    heatmap: str = Field(description="PNG data URI of the heatmap on its own")


class ImagesOut(BaseModel):
    source: str
    model_input: str


class AnalyzeResponse(BaseModel):
    prediction: PredictionOut
    scores: list[ScoreOut]
    explained_class: ExplainedClassOut
    explanations: list[ExplanationOut]
    images: ImagesOut
    warnings: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    detail: str

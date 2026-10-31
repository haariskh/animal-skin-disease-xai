"""HTTP routes."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from aniderm.explain.registry import AVAILABLE_METHODS, method_catalogue
from aniderm.utils.imaging import ImageDecodeError, load_image
from aniderm.utils.logging import get_logger
from api.dependencies import get_config, get_pipeline, pipeline_status
from api.schemas import AnalyzeResponse, HealthResponse, MetadataResponse

logger = get_logger("api.routes")
router = APIRouter(prefix="/api")

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
}


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    loaded, detail = pipeline_status()
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        detail=detail,
    )


@router.get("/metadata", response_model=MetadataResponse, tags=["system"])
def metadata() -> MetadataResponse:
    """Everything the UI needs to render itself before any image is uploaded."""
    config = get_config()
    try:
        pipeline = get_pipeline()
    except FileNotFoundError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    described = pipeline.predictor.describe()
    return MetadataResponse(
        backbone=described["backbone"],
        image_size=described["image_size"],
        dataset=described["dataset"],
        trained_at=described["trained_at"],
        num_classes=described["num_classes"],
        device=described["device"],
        classes=described["classes"],
        methods=method_catalogue(config.xai),
        metrics=described["metrics"],
        low_confidence_threshold=config.serving.low_confidence_threshold,
        max_upload_mb=config.serving.max_upload_mb,
    )


@router.post("/analyze", response_model=AnalyzeResponse, tags=["inference"])
async def analyze(
    file: UploadFile = File(..., description="The image to classify"),
    methods: str | None = Form(
        None, description="Comma-separated explainer names; omit for all enabled ones"
    ),
    target_class: str | None = Form(
        None, description="Explain this class instead of the top prediction"
    ),
    include_faithfulness: str | None = Form(
        None,
        description="'true' or 'false'; omit to use the value from the config",
    ),
) -> AnalyzeResponse:
    """Classify one image and explain the prediction."""
    config = get_config()
    try:
        pipeline = get_pipeline()
    except FileNotFoundError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{file.content_type}'. Send a JPEG, PNG, WebP or BMP.",
        )

    payload = await file.read()
    if not payload:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="The uploaded file is empty.")
    if len(payload) > config.serving.max_upload_bytes:
        raise HTTPException(
            # Spelled numerically because Starlette renamed this constant and the
            # supported FastAPI range spans both names.
            413,
            detail=(
                f"Image is {len(payload) / 1024 / 1024:.1f} MB; the limit is "
                f"{config.serving.max_upload_mb} MB."
            ),
        )

    try:
        image = load_image(payload)
    except ImageDecodeError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    # Argument validation happens before the try block: a 400 raised inside it would be
    # swallowed by the catch-all below and re-reported as a 500.
    selected = _parse_methods(methods)
    want_faithfulness = _parse_bool(include_faithfulness)

    try:
        result = pipeline.analyse(
            image=image,
            methods=selected,
            target_class=target_class,
            include_faithfulness=want_faithfulness,
        )
    except KeyError as exc:
        # The pipeline raises KeyError for an unknown class or a disabled method, both
        # of which are the caller's mistake rather than a server fault.
        message = exc.args[0] if exc.args else str(exc)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(message)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Analysis failed")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {exc}",
        ) from exc

    return AnalyzeResponse(**result.to_dict())


def _parse_bool(raw: str | None) -> bool | None:
    """Read a boolean form field, returning None when the client omitted it.

    This is parsed by hand rather than declared as ``bool = Form(True)`` because an
    omitted boolean form field resolves to False in the FastAPI version this project
    pins, silently ignoring the declared default. String fields are unaffected, so the
    value is taken as text and converted here. Returning None lets the pipeline fall
    back to the config rather than guessing.
    """
    if raw is None:
        return None
    lowered = raw.strip().lower()
    if lowered in {"true", "1", "yes", "on"}:
        return True
    if lowered in {"false", "0", "no", "off"}:
        return False
    raise HTTPException(
        status.HTTP_400_BAD_REQUEST,
        detail=f"include_faithfulness must be true or false, got '{raw}'",
    )


def _parse_methods(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    names = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [name for name in names if name not in AVAILABLE_METHODS]
    if unknown:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unknown method(s): {', '.join(unknown)}. "
                f"Available: {', '.join(sorted(AVAILABLE_METHODS))}"
            ),
        )
    return names or None

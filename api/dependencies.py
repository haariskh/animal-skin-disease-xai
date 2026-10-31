"""Application state: configuration and the lazily-loaded analysis pipeline.

The model is loaded once, on the first request that needs it rather than at import time,
so ``uvicorn`` still starts (and ``/api/health`` still answers) when no checkpoint exists
yet. A freshly cloned repository has no trained model, and a server that refuses to boot
in that state is needlessly hostile.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from aniderm.config import Config, resolve_path
from aniderm.inference.pipeline import AnalysisPipeline, build_pipeline
from aniderm.utils.logging import get_logger

logger = get_logger("api.deps")

_lock = threading.Lock()
_config: Config | None = None
_pipeline: AnalysisPipeline | None = None
_load_error: str | None = None


def config_path() -> Path:
    return resolve_path(os.environ.get("ANIDERM_CONFIG", "configs/default.yaml"))


def checkpoint_path(config: Config) -> Path:
    override = os.environ.get("ANIDERM_CHECKPOINT")
    return resolve_path(override) if override else config.serving.checkpoint


def get_config() -> Config:
    global _config
    with _lock:
        if _config is None:
            _config = Config.load(config_path())
            logger.info("Config loaded from %s", config_path())
        return _config


def get_pipeline() -> AnalysisPipeline:
    """Return the shared pipeline, loading the checkpoint on first use.

    Raises ``FileNotFoundError`` when no checkpoint exists so the route layer can turn
    that into a 503 with instructions, rather than a bare 500.
    """
    global _pipeline, _load_error
    config = get_config()

    with _lock:
        if _pipeline is not None:
            return _pipeline

        path = checkpoint_path(config)
        if not path.exists():
            _load_error = (
                f"No checkpoint at {path}. Train a model first:\n"
                f"  python scripts/prepare_data.py\n"
                f"  python scripts/train.py"
            )
            raise FileNotFoundError(_load_error)

        _pipeline = build_pipeline(
            checkpoint=path,
            xai=config.xai,
            device_preference=config.training.device,
            low_confidence_threshold=config.serving.low_confidence_threshold,
            seed=config.project.seed,
        )
        _load_error = None
        logger.info("Analysis pipeline ready")
        return _pipeline


def pipeline_status() -> tuple[bool, str]:
    """Report loadability without raising, for the health endpoint."""
    try:
        get_pipeline()
        return True, "Model loaded and ready."
    except Exception as exc:
        return False, str(exc)


def reset_state() -> None:
    """Drop the cached config and pipeline. Used by the tests."""
    global _config, _pipeline, _load_error
    with _lock:
        _config = None
        _pipeline = None
        _load_error = None

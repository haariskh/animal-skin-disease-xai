"""FastAPI application entry point.

Run it with:

    uvicorn api.main:app --reload --port 8000

Interactive API docs are then at http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Make `src/` importable when the server is launched from the repository root, so no
# editable install or PYTHONPATH fiddling is needed to get started.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from aniderm.utils.logging import get_logger, setup_logging  # noqa: E402
from api.dependencies import get_config, pipeline_status  # noqa: E402
from api.routes import router  # noqa: E402

logger = get_logger("api")

FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    config = get_config()
    logger.info("Starting %s", config.project.name)

    # Warm the model at boot so the first upload is not slowed by loading weights. A
    # missing checkpoint is only logged: the server still serves /api/health, which is
    # how the UI knows to tell the user to train a model.
    loaded, detail = pipeline_status()
    if loaded:
        logger.info("Model ready")
    else:
        logger.warning("Starting without a model. %s", detail)
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Animal Skin Disease Classification with Explainable AI",
    description=(
        "Classifies an animal skin photograph into a dermatological category and explains "
        "which parts of the image drove that decision, using Grad-CAM++, Integrated "
        "Gradients and LIME."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

_config = get_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(_config.serving.allowed_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/api", include_in_schema=False)
def api_root() -> JSONResponse:
    return JSONResponse(
        {
            "name": "aniderm",
            "docs": "/docs",
            "endpoints": ["/api/health", "/api/metadata", "/api/analyze"],
        }
    )


# When the frontend has been built (`npm run build`), serve it from the same origin so
# the whole application runs on one port and CORS stops mattering. During development
# the Vite dev server on :5173 proxies to this API instead.
if FRONTEND_DIST.is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_DIST / "assets"),
        name="assets",
    )

    @app.get("/", include_in_schema=False)
    def serve_index() -> FileResponse:
        return FileResponse(FRONTEND_DIST / "index.html")

else:

    @app.get("/", include_in_schema=False)
    def no_frontend() -> JSONResponse:
        return JSONResponse(
            {
                "message": "API is running, but the frontend has not been built.",
                "build_it": "cd frontend && npm install && npm run build",
                "or_develop": "cd frontend && npm run dev   (then open http://localhost:5173)",
                "api_docs": "/docs",
            }
        )


if __name__ == "__main__":
    import uvicorn

    config = get_config()
    uvicorn.run(
        "api.main:app",
        host=os.environ.get("ANIDERM_HOST", config.serving.host),
        port=int(os.environ.get("ANIDERM_PORT", config.serving.port)),
        reload=True,
    )

"""The server must still start, and say something useful, with no trained model."""

from __future__ import annotations

import pytest
import yaml
from fastapi.testclient import TestClient


@pytest.fixture
def client_without_model(monkeypatch, tmp_path, dataset_config_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset_config": str(dataset_config_path),
                "model": {"backbone": "resnet18", "pretrained": False, "image_size": 64},
                "training": {"device": "cpu", "num_workers": 0},
                "serving": {"checkpoint": str(tmp_path / "nothing-here.pt")},
            }
        )
    )
    monkeypatch.setenv("ANIDERM_CONFIG", str(config_path))
    monkeypatch.setenv("ANIDERM_CHECKPOINT", str(tmp_path / "nothing-here.pt"))

    import api.dependencies as deps

    deps.reset_state()
    from api.main import app

    with TestClient(app) as client:
        yield client
    deps.reset_state()


def test_health_is_degraded_but_answers(client_without_model) -> None:
    body = client_without_model.get("/api/health").json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
    assert "train.py" in body["detail"]


def test_metadata_is_503_with_instructions(client_without_model) -> None:
    response = client_without_model.get("/api/metadata")
    assert response.status_code == 503
    assert "prepare_data.py" in response.json()["detail"]


def test_analyze_is_503_rather_than_a_crash(client_without_model) -> None:
    response = client_without_model.post(
        "/api/analyze", files={"file": ("x.jpg", b"\xff\xd8\xff", "image/jpeg")}
    )
    assert response.status_code == 503

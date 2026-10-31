"""HTTP surface: status codes, validation and response shape."""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def client(monkeypatch, tmp_path, checkpoint_path, config, dataset_config_path):
    """A TestClient wired to the fixture config and the fixture checkpoint."""
    import yaml

    # The API reads its config from the environment, so point it at the fixtures.
    config_path = tmp_path / "api_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project": {"seed": 11, "artifacts_dir": str(tmp_path / "artifacts")},
                "dataset_config": str(dataset_config_path),
                "model": {"backbone": "resnet18", "pretrained": False, "image_size": 64},
                "training": {"device": "cpu", "num_workers": 0, "amp": False},
                "xai": {
                    "default_method": "gradcam_pp",
                    "enabled_methods": ["gradcam_pp", "integrated_gradients"],
                    "integrated_gradients": {"steps": 8, "internal_batch_size": 4},
                    "faithfulness": {"enabled": True, "steps": 3},
                },
                "serving": {"checkpoint": str(checkpoint_path), "max_upload_mb": 1},
            }
        )
    )
    monkeypatch.setenv("ANIDERM_CONFIG", str(config_path))
    monkeypatch.setenv("ANIDERM_CHECKPOINT", str(checkpoint_path))

    import api.dependencies as deps

    deps.reset_state()
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client
    deps.reset_state()


def _image_bytes(size=(90, 70), fmt="JPEG") -> bytes:
    array = np.random.default_rng(4).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format=fmt)
    return buffer.getvalue()


def test_health_reports_a_loaded_model(client) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_metadata_describes_the_model_and_the_methods(client) -> None:
    body = client.get("/api/metadata").json()
    assert body["backbone"] == "resnet18"
    assert body["image_size"] == 64
    assert [c["key"] for c in body["classes"]] == [
        "bacterial_dermatosis",
        "fungal_infections",
        "healthy",
    ]
    assert {m["name"] for m in body["methods"]} == {"gradcam_pp", "integrated_gradients"}
    assert sum(m["is_default"] for m in body["methods"]) == 1


def test_analyze_returns_the_full_payload(client) -> None:
    response = client.post(
        "/api/analyze",
        files={"file": ("photo.jpg", _image_bytes(), "image/jpeg")},
        data={"methods": "gradcam_pp"},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["prediction"]["class_key"] in {
        "bacterial_dermatosis",
        "fungal_infections",
        "healthy",
    }
    assert len(body["scores"]) == 3
    assert body["explanations"][0]["overlay"].startswith("data:image/png;base64,")
    assert body["explanations"][0]["regions"]["summary"]
    assert body["images"]["source"].startswith("data:image/png;base64,")


def test_faithfulness_runs_when_the_field_is_omitted(client) -> None:
    """Regression guard: an omitted boolean form field must not silently become False."""
    body = client.post(
        "/api/analyze",
        files={"file": ("photo.jpg", _image_bytes(), "image/jpeg")},
        data={"methods": "gradcam_pp"},
    ).json()
    assert "insertion_auc" in body["explanations"][0]["faithfulness"]


def test_faithfulness_can_be_turned_off_per_request(client) -> None:
    body = client.post(
        "/api/analyze",
        files={"file": ("photo.jpg", _image_bytes(), "image/jpeg")},
        data={"methods": "gradcam_pp", "include_faithfulness": "false"},
    ).json()
    assert body["explanations"][0]["faithfulness"] == {}


def test_all_enabled_methods_run_when_none_are_named(client) -> None:
    body = client.post(
        "/api/analyze", files={"file": ("photo.jpg", _image_bytes(), "image/jpeg")}
    ).json()
    assert {item["method"] for item in body["explanations"]} == {
        "gradcam_pp",
        "integrated_gradients",
    }


def test_png_uploads_are_accepted(client) -> None:
    response = client.post(
        "/api/analyze",
        files={"file": ("photo.png", _image_bytes(fmt="PNG"), "image/png")},
        data={"methods": "gradcam_pp"},
    )
    assert response.status_code == 200


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"methods": "telepathy"}, 400),
        ({"target_class": "eczema"}, 400),
        ({"include_faithfulness": "perhaps"}, 400),
    ],
)
def test_bad_arguments_are_client_errors_not_server_errors(client, payload, expected) -> None:
    response = client.post(
        "/api/analyze",
        files={"file": ("photo.jpg", _image_bytes(), "image/jpeg")},
        data=payload,
    )
    assert response.status_code == expected
    assert response.json()["detail"]


def test_undecodable_file_is_a_400(client) -> None:
    response = client.post(
        "/api/analyze", files={"file": ("photo.jpg", b"definitely not an image", "image/jpeg")}
    )
    assert response.status_code == 400
    assert "decode" in response.json()["detail"].lower()


def test_empty_file_is_a_400(client) -> None:
    response = client.post("/api/analyze", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert response.status_code == 400


def test_wrong_content_type_is_a_415(client) -> None:
    response = client.post(
        "/api/analyze", files={"file": ("notes.pdf", b"%PDF-1.4", "application/pdf")}
    )
    assert response.status_code == 415


def test_oversized_upload_is_a_413(client) -> None:
    big = b"\xff\xd8\xff" + b"0" * (2 * 1024 * 1024)
    response = client.post("/api/analyze", files={"file": ("big.jpg", big, "image/jpeg")})
    assert response.status_code == 413
    assert "limit" in response.json()["detail"]


def test_explaining_an_alternative_class(client) -> None:
    body = client.post(
        "/api/analyze",
        files={"file": ("photo.jpg", _image_bytes(), "image/jpeg")},
        data={"methods": "gradcam_pp", "target_class": "healthy"},
    ).json()
    assert body["explained_class"]["key"] == "healthy"


def test_openapi_schema_is_generated(client) -> None:
    schema = client.get("/openapi.json").json()
    assert "/api/analyze" in schema["paths"]

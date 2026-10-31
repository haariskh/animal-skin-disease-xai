"""Configuration loading and the validation that stops bad configs reaching training."""

from __future__ import annotations

import pytest
import yaml

from aniderm.config import ClassSpec, Config, DatasetConfig, SplitConfig


def test_loads_full_tree(config: Config) -> None:
    assert config.dataset.num_classes == 3
    assert config.dataset.class_keys == [
        "bacterial_dermatosis",
        "fungal_infections",
        "healthy",
    ]
    assert config.model.pretrained is False
    assert config.xai.default_method == "gradcam_pp"


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("Bacterial_dermatosis", 0),
        ("bacterial dermatosis", 0),
        ("BACTERIALDERMATOSIS", 0),
        ("ringworm", 1),  # an alias that is not the display name
        ("Normal", 2),
    ],
)
def test_class_lookup_is_punctuation_and_case_insensitive(
    dataset_config: DatasetConfig, given: str, expected: int
) -> None:
    assert dataset_config.index_of(given) == expected


def test_unknown_class_raises(dataset_config: DatasetConfig) -> None:
    with pytest.raises(KeyError):
        dataset_config.index_of("psoriasis")


def test_duplicate_alias_across_classes_is_rejected(tmp_path, dataset_config_path) -> None:
    payload = yaml.safe_load(dataset_config_path.read_text())
    payload["classes"][1]["aliases"].append("Bacterial-Dermatosis")
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(payload))

    with pytest.raises(ValueError, match="claimed by both"):
        DatasetConfig.load(path)


def test_class_spec_match_keys_normalise() -> None:
    spec = ClassSpec(
        key="fungal_infections",
        display_name="Fungal Infections",
        aliases=("Ring-worm",),
    )
    assert "fungalinfections" in spec.match_keys
    assert "ringworm" in spec.match_keys


@pytest.mark.parametrize("val,test", [(0.0, 0.2), (0.2, 1.5), (0.5, 0.5)])
def test_impossible_split_sizes_are_rejected(val: float, test: float) -> None:
    with pytest.raises(ValueError):
        SplitConfig(val_size=val, test_size=test)


def test_invalid_group_pattern_is_rejected() -> None:
    with pytest.raises(ValueError, match="not a valid regex"):
        SplitConfig(group_pattern="([unclosed")


def test_unknown_monitor_is_rejected(tmp_path, dataset_config_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "dataset_config": str(dataset_config_path),
                "training": {"monitor": "val_vibes"},
            }
        )
    )
    with pytest.raises(ValueError, match="training.monitor"):
        Config.load(path)


def test_default_xai_method_must_be_enabled(tmp_path, dataset_config_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "dataset_config": str(dataset_config_path),
                "xai": {"default_method": "lime", "enabled_methods": ["gradcam_pp"]},
            }
        )
    )
    with pytest.raises(ValueError, match="not listed in enabled_methods"):
        Config.load(path)


def test_shipped_configs_all_load() -> None:
    """The configs in the repository must stay loadable as the schema changes."""
    from aniderm.config import REPO_ROOT

    for path in sorted((REPO_ROOT / "configs" / "datasets").glob("*.yaml")):
        dataset = DatasetConfig.load(path)
        assert dataset.num_classes >= 2
        assert len(set(dataset.class_keys)) == dataset.num_classes

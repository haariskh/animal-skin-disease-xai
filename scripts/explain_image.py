#!/usr/bin/env python3
"""Classify and explain one image from the command line.

    python scripts/explain_image.py path/to/photo.jpg
    python scripts/explain_image.py photo.jpg --methods gradcam_pp integrated_gradients
    python scripts/explain_image.py photo.jpg --target-class healthy   # counterfactual

The last form is worth knowing about: forcing the explanation onto a class the model did
NOT pick shows you where the evidence for that alternative would have had to be, which is
often how you catch the model looking at the background.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aniderm.config import Config  # noqa: E402
from aniderm.explain.registry import AVAILABLE_METHODS  # noqa: E402
from aniderm.inference.pipeline import build_pipeline  # noqa: E402
from aniderm.utils.imaging import load_image  # noqa: E402
from aniderm.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger("explain")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image", help="path to the image to analyse")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--methods", nargs="+", default=None, choices=sorted(AVAILABLE_METHODS),
        help="which explainers to run (default: all enabled in the config)",
    )
    parser.add_argument(
        "--target-class", default=None,
        help="explain this class instead of the top prediction",
    )
    parser.add_argument("--output-dir", default="artifacts/reports/explanations")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--no-faithfulness", action="store_true", help="skip the deletion/insertion check"
    )
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()
    config = Config.load(args.config)

    image_path = Path(args.image)
    if not image_path.exists():
        logger.error("Image not found: %s", image_path)
        return 1

    pipeline = build_pipeline(
        checkpoint=args.checkpoint or config.serving.checkpoint,
        xai=config.xai,
        device_preference=args.device or config.training.device,
        low_confidence_threshold=config.serving.low_confidence_threshold,
        seed=config.project.seed,
    )

    image = load_image(image_path)
    result = pipeline.analyse(
        image=image,
        methods=args.methods,
        target_class=args.target_class,
        include_faithfulness=not args.no_faithfulness,
    )

    print("=" * 78)
    print(f"{image_path.name}")
    print("=" * 78)
    print(f"Prediction : {result.prediction.top.display_name} "
          f"({result.prediction.top.probability:.1%})")
    print(f"Explaining : {result.explained_class_display_name}")
    print()
    print("All classes")
    for score in result.prediction.scores:
        bar = "#" * int(round(score.probability * 40))
        print(f"  {score.display_name:<40} {score.probability:>6.1%}  {bar}")

    for explanation in result.explanations:
        print()
        print(f"-- {explanation.method_label} ({explanation.runtime_ms:.0f} ms)")
        print(f"   {explanation.regions['summary']}")
        concentration = explanation.regions["concentration"]
        print(f"   focus: top 10% of pixels hold "
              f"{concentration['top_10pct_share']:.0%} of the evidence")
        if explanation.faithfulness and "error" not in explanation.faithfulness:
            faith = explanation.faithfulness
            print(f"   faithfulness: deletion AUC {faith['deletion_auc']:.3f} "
                  f"(lower better), insertion AUC {faith['insertion_auc']:.3f} (higher better)")
            if "random_insertion_auc" in faith:
                print(f"                 random control insertion AUC "
                      f"{faith['random_insertion_auc']:.3f}")
            print(f"   verdict: {faith['verdict']}")

    for warning in result.warnings:
        print(f"\n[!] {warning}")

    output_dir = Path(args.output_dir) / image_path.stem
    _write_outputs(result, output_dir)
    print(f"\nImages and report written to {output_dir}")
    return 0


def _write_outputs(result, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    def write_data_uri(data_uri: str, path: Path) -> None:
        _, _, encoded = data_uri.partition("base64,")
        path.write_bytes(base64.b64decode(encoded))

    write_data_uri(result.source_data_uri, output_dir / "00_source.png")
    write_data_uri(result.model_input_data_uri, output_dir / "01_model_input.png")
    for explanation in result.explanations:
        write_data_uri(
            explanation.overlay_data_uri, output_dir / f"{explanation.method}_overlay.png"
        )
        write_data_uri(
            explanation.heatmap_data_uri, output_dir / f"{explanation.method}_heatmap.png"
        )

    payload = result.to_dict()
    # The data URIs are large and already on disk as PNGs; drop them from the JSON.
    payload["images"] = {"note": "written alongside this file as PNG"}
    for item in payload["explanations"]:
        item.pop("overlay", None)
        item.pop("heatmap", None)
    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())

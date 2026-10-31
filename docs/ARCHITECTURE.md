# Architecture

## Layout

```
animal-skin-disease-xai/
├── configs/
│   ├── default.yaml              everything: model, training, augmentation, XAI, serving
│   └── datasets/                 one file per dataset; swapping these swaps the problem
├── scripts/                      the CLI entry points, in the order you run them
│   ├── download_dataset.py
│   ├── prepare_data.py
│   ├── make_synthetic_dataset.py pipeline smoke test without a download
│   ├── train.py
│   ├── evaluate.py
│   └── explain_image.py
├── src/aniderm/                  the importable library; no script logic lives here
│   ├── config.py                 typed config tree, validated at load time
│   ├── data/                     discovery → manifest → Dataset → DataLoader
│   ├── models/                   backbone factory, checkpointing, layer introspection
│   ├── training/                 loop, losses, metrics, cross-validation
│   ├── explain/                  Grad-CAM++, IG, LIME, regions, faithfulness
│   ├── inference/                predictor and the predict→explain→render pipeline
│   └── utils/                    seeding, logging, device, imaging
├── api/                          FastAPI: schemas, dependencies, routes, app
├── frontend/                     React + Vite single page
├── tests/                        131 tests, no network or dataset required
├── data/                         raw downloads and generated manifests (git-ignored)
└── artifacts/                    checkpoints and reports (git-ignored)
```

## How a request flows

```
upload ──> api/routes.py            validate type, size, decode
            │
            └─> AnalysisPipeline.analyse()
                 ├─ Predictor.predict()         tensor → logits → sorted probabilities
                 ├─ for each method:
                 │    ├─ build_explainer()      from config
                 │    ├─ explainer.explain()    raw saliency, possibly 7×7
                 │    ├─ resize + normalise     → input resolution, [0,1]
                 │    ├─ analyse_regions()      grid, hotspots, concentration, summary
                 │    ├─ deletion_insertion()   faithfulness + random control
                 │    └─ overlay_heatmap()      → PNG data URI
                 └─ confidence warnings
```

Two decisions in there are load-bearing.

Normalisation and resizing happen once, centrally, after the explainer returns. Grad-CAM++
produces a feature-map-sized grid while the pixel methods produce full resolution, and
every stage downstream — regions, faithfulness, overlay — needs one geometry to be
comparable. Pushing that into each explainer would mean three chances to get it subtly
different.

The overlay is drawn on the square-resized view the model actually received, not on the
original photograph. Compositing a 224×224 saliency map onto a 4000×3000 phone photo of a
different aspect ratio shifts every highlighted pixel relative to the skin underneath it.
The original upload is returned separately as `images.source`.

A failing explainer is caught per method. The response still carries the prediction and
whichever explanations succeeded, with the failure listed in `warnings`.

## Configuration

One YAML tree, loaded into frozen dataclasses that validate on construction. Mistakes that
would otherwise surface as a confusing error twenty minutes into training fail at load:
a `monitor` that is not a real metric, split fractions that leave no training data, a
`default_method` that is not in `enabled_methods`, two classes claiming the same alias.

Nothing under `src/aniderm/` hardcodes a class name or a class count. Switching dataset is
one line in `configs/default.yaml` plus a re-run of `prepare_data.py`.

CLI flags override the YAML without writing to it, through `dataclasses.replace`, so a
quick experiment never leaves the config file in a state you have to remember to undo.

## The manifest

`prepare_data.py` walks the download, resolves each image's class, assigns splits, and
writes one CSV: `path, class_key, label, group, split, source_split`. Every later stage
reads that file rather than re-deriving a split, which is what makes a training run, an
evaluation and an XAI report comparable to each other.

Paths are stored relative to the repository root where possible, so the manifest survives
the project being moved.

`load_manifest` verifies that the stored labels still agree with the dataset config before
returning. Editing the class order in YAML after training and then evaluating against the
old manifest would map predictions to the wrong disease, silently, and every number after
that would be wrong in a way no metric would reveal. That check turns it into an error.

## Training

Two stages, which is the standard recipe for a few hundred images.

First the backbone is frozen and only the classifier head trains. A randomly initialised
head produces large gradients in its first steps, and letting those flow into pretrained
weights damages the features the whole approach depends on.

Then everything unfreezes, with the backbone on a learning rate an order of magnitude
below the head, warmup for a couple of epochs, and cosine decay after.

Checkpointing and early stopping follow macro-F1, not accuracy. On an imbalanced
four-class set a model that gives up on the smallest class can still post a respectable
accuracy, and that is the failure this project most needs to avoid.

## Checkpoints

A checkpoint carries its own metadata: backbone name, image size, the full class list with
display names and colours, the resolved Grad-CAM target layer, and the validation metrics
it was selected on.

Serving therefore never reads the dataset YAML. A config file that drifted after training
would otherwise produce a model that maps output index 2 to the wrong disease while
looking entirely healthy, which is the worst available failure for this project. The state
dict is loaded strictly, so a mismatched architecture or class count is an error rather
than a quietly broken model.

## Grad-CAM target layers

Rather than hardcoding a path per architecture — `layer4` for ResNet, `conv_head` for
EfficientNet, and so on — the target layer is found by running one dummy forward pass with
hooks on every leaf module and keeping the last one whose output is a 4-D tensor larger
than 1×1 spatially.

That is why `model.backbone` can be any timm name in the config and the XAI layer keeps
working. On an EfficientNet-B0 it resolves to `bn2.act`; on a ResNet-18, to the final
block's activation. The resolved name is saved into the checkpoint, so serving does not
repeat the search.

## Frontend

React and Vite, one page, no component library. State is held in `App.tsx` and passed down;
there is not enough of it to justify anything more.

Requests go to a relative `/api` path, which means one build works two ways: behind the
Vite dev proxy on port 5173 during development, and served directly by uvicorn from
`frontend/dist` in production, where same-origin makes CORS irrelevant.

Images arrive as base64 data URIs in the JSON response. The server needs no writable static
directory, no cache eviction, and no second round trip.

## Tests

131 tests, none of which need the dataset, a trained model, or network access. Fixtures
generate a small dataset in a temp directory and save a checkpoint of an untrained model.

The tests worth knowing about are the ones that check correctness rather than plumbing.
`test_gradcam_finds_the_patch_the_model_actually_uses` builds a model whose class score
provably depends on one quadrant of the image, then asserts the heatmap points there —
shape assertions alone would pass on a method returning noise.
`test_no_group_appears_in_two_splits` guards the leakage property that makes the reported
accuracy mean anything. `test_loading_a_manifest_with_shifted_labels_is_refused` guards the
silent-mislabelling failure above.

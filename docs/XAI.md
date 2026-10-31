# Explainability

Three methods ship here. Grad-CAM++ is the default, Integrated Gradients and LIME are a
click away, and every map is scored for whether it can be believed.

## Why Grad-CAM++ is the default

The brief asked for one appropriate method. For a convolutional classifier serving
interactive uploads, Grad-CAM++ wins on four counts.

It explains the deployed network. The map is built from the gradient of the predicted
class with respect to the last convolutional feature map, so it reflects what the model
computed. LIME, by contrast, fits a separate linear model to sampled perturbations and
explains *that*; the surrogate can be a good approximation or a poor one, and nothing in
the output tells you which.

It costs one forward and one backward pass — about 60–100 ms on CPU for a ResNet-18 at
128px, faster on a GPU. LIME needs several hundred forward passes (8.6 seconds in the
same setting). KernelSHAP needs more again. For a UI where someone drops in a photo and
waits, that difference decides whether the explanation is part of the interaction or a
separate job.

It is deterministic. The same image gives the same map every time, so two runs that
disagree mean something changed. LIME samples, and although the seed is fixed here for
reproducibility, its result still shifts with `num_samples`.

And over plain Grad-CAM it fixes a specific failure. Grad-CAM averages the gradient across
each channel into a single scalar weight. When a class is present in several disconnected
places — which is what multiple lesions on one animal look like — that averaging
collapses the evidence onto the strongest site and the others fade out. Grad-CAM++ weights
each spatial position separately:

```
α_ij = g_ij² / (2·g_ij² + (Σ_ab A_ab)·g_ij³)
w_k  = Σ_ij α_ij · relu(g_ij)
cam  = relu(Σ_k w_k · A_k)
```

so scattered lesions all survive into the map. The closing ReLU matters too: negative
contributions are evidence *against* the class, and letting them through would paint a
region that argues against the prediction as though it supported it.

Chattopadhay, Sarkar, Howlader and Balasubramanian, *Grad-CAM++: Generalized
Gradient-Based Visual Explanations for Deep Convolutional Networks*, WACV 2018.

## Why the other two are still here

**Integrated Gradients** answers a different question. Grad-CAM++ works at feature-map
resolution — 7×7 for an EfficientNet-B0 at 224px, upsampled for display — so it says
"this region", not "these pixels". Integrated Gradients accumulates gradients along a
straight path from a reference image to the real one and attributes at full resolution,
with a completeness guarantee: the attributions sum to the difference in model output
between the image and the baseline. When a lesion border is what distinguishes two
classes, that resolution is the difference between a useful explanation and a blob.

Sundararajan, Taly and Yan, *Axiomatic Attribution for Deep Networks*, ICML 2017.

**LIME** is the only one of the three that never looks inside the model. It segments the
image into superpixels, switches groups of them off, watches the prediction move, and fits
a sparse linear model to that behaviour. Because it shares no machinery with the gradient
methods, agreement between them is corroboration rather than two views of the same
computation. Disagreement is a signal worth chasing.

Ribeiro, Singh and Guestrin, *"Why Should I Trust You?": Explaining the Predictions of Any
Classifier*, KDD 2016.

SHAP was considered and left out. KernelSHAP on images is LIME's cost multiplied again,
and DeepSHAP's assumptions are a poor fit for a model with this much nonlinearity in the
head. Nothing it adds here justifies the wait.

## The part most projects skip

A saliency map can look convincing and still be wrong. Nothing about a pretty heatmap
proves the model used those pixels, and the eye is easily satisfied by a blob that happens
to sit on a lesion.

So every explanation is scored by a deletion/insertion test. Rank the pixels by attributed
importance, then either remove them from the image or add them to a reference, measuring
the predicted probability at each step.

- **Deletion** replaces the most important pixels first with flat neutral grey. An honest
  map makes confidence collapse quickly, so a **low** area under the curve is good.
- **Insertion** reveals the most important pixels first against a blurred copy of the
  image. An honest map makes confidence recover quickly, so a **high** area is good.

A random ranking is scored alongside as a control, and the verdict reported in the UI
compares against it. A map that barely beats random is reported as such, in those words.

Petsiuk, Das and Saenko, *RISE: Randomized Input Sampling for Explanation of Black-box
Models*, BMVC 2018.

### Why the two curves use different references

This caught a real bug during development, and it is worth understanding before changing
the config.

Running both curves against a blurred reference made every method score the same as
random. Blur removes texture but preserves colour, and colour is a large part of what a
dermatology model keys on — erythema is redness. "Deleting" a red lesion into its blurred
self leaves the red behind, the prediction does not move, and the test concludes that no
explanation explains anything.

Deletion therefore uses a flat neutral reference (`zero`, which is the ImageNet mean
colour once normalisation is undone) and insertion uses blur, as in the original method.
Switching deletion to blur in `configs/default.yaml` will make the scores collapse toward
random again, and that is the config's fault rather than the explainer's.

Note that `zero` is not black. Zero in normalised tensor space is the dataset mean, a mid
grey. A genuinely black baseline systematically understates dark pixels, which is a real
problem on photographs of dark fur or shadowed skin, so `black` is available but is not
the default anywhere.

## Reading the output

Beyond the heatmap itself, each explanation carries numbers.

`top_10pct_share` is the fraction of all evidence held by the most salient tenth of the
pixels. Near 1.0 means the model pinned a specific area. Near 0.1 means the attribution is
spread evenly and the explanation is not saying much — the UI says so in words rather than
leaving a diffuse heatmap to imply precision it does not have.

Hotspots are the connected regions above half intensity, each reported with its share of
the total evidence, the fraction of the frame it covers, and where in the image it sits.
Specks smaller than a 2000th of the image are dropped as interpolation noise.

The 3×3 grid gives each ninth of the frame its share of the evidence. It is the fastest way
to catch a model that has learned the background: a confident prediction whose evidence
sits in the corners is reading the examination table, not the animal.

## Explaining a class the model did not pick

Both the UI and the CLI can force the explanation onto any class.

```bash
python scripts/explain_image.py photo.jpg --target-class healthy
```

This asks where the evidence for that alternative would have had to be. It is often how
you catch a model looking at the wrong thing, because a counterfactual map that lands on
the same region as the top-class map means the model is not distinguishing the two
classes spatially at all.

## Limits

Grad-CAM++ needs a convolutional feature map. On a vision transformer the target-layer
resolution will fail with a message saying so, and Integrated Gradients or LIME should be
used instead.

Every map explains *this model on this image*. A trustworthy-looking explanation of a
wrong prediction is still a wrong prediction, and the faithfulness score measures whether
the map describes the model, not whether the model is right.

LIME's superpixels come from quickshift, which segments poorly on smooth, low-contrast
images. On uniform skin it tends to produce large, uninformative segments — visible in the
test runs as a worse localisation score than either gradient method. The segmentation
parameters are exposed in the config for that reason.

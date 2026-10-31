# Dataset

## What this project trains on

**Kaggle — `yashmotiani/dogs-skin-disease-dataset`**, 443 RGB photographs across four
diagnostic categories.

| Class | Images |
|---|---|
| Fungal infections | 137 |
| Healthy | 119 |
| Bacterial dermatosis | 97 |
| Hypersensitivity dermatitis | 90 |

Those per-class counts are not guessed from the Kaggle page, which does not publish them.
They come from a peer-reviewed study that used this exact dataset and reported its
composition: Yousef et al. (2025), *Mobile-based deep learning approach for the early
diagnosis of canine skin and eye diseases using MobileNetV2*, which cites the dataset as
Motiani (2024) and lists 443 skin images with the breakdown above. That study trained
MobileNetV2 at 128×128 for 25 epochs and reported 93.22% training / 86.31% validation
accuracy on a combined skin-and-eye set, with ResNet50 at 80.15% and InceptionV3 at
67.26% on the same test set.

Verify the counts yourself after downloading:

```bash
python scripts/prepare_data.py --inspect-only
```

If what lands on disk differs from the table above, the dataset was re-uploaded. Edit
`configs/datasets/dogs_skin_4class.yaml` to match and re-run.

## Where the four classes come from

The taxonomy — bacterial dermatosis, fungal infections, hypersensitivity allergic
dermatosis, healthy — was defined by Hwang, Shin, Park, Kwon and Kang (2022),
*Classification of dog skin diseases using deep learning with images captured from a
multispectral imaging device*, Molecular & Cellular Toxicology 18, published by Springer
(doi:10.1007/s13273-022-00249-7). They imaged 95 dogs at a veterinary hospital: 23 with
bacterial dermatosis, 19 with fungal infections, 23 with hypersensitivity allergic
dermatosis, and 30 healthy.

## A correction worth recording

That paper has a companion Mendeley Data record,
[doi:10.17632/5dbht54kw7.1](https://data.mendeley.com/datasets/5dbht54kw7/1), released
under CC BY 4.0 and titled *Classification of pet dog skin diseases using deep learning
with images captured from multispectral imaging device*. Its title and licence make it
look like the obvious thing to train on.

It contains no images.

The record has one version, published 2022-02-04, holding a single file: `image_label.txt`,
2,379 bytes, 62 lines pairing an image id with a label (26 Healthy, 13
Hypersensitivity_allergic_dermatosis, 12 Bacterial_dermatosis, 11 Fungal_infections). The
multispectral captures behind the paper were never deposited. Anyone citing that DOI as
their training data has not downloaded it.

So the Kaggle mirror is the image source, and the Springer paper is what you cite for the
class definitions.

## Alternatives, and why they are not the default

**Roboflow Universe — `dog-skin-disease-dataset`.** Same four classes, CC BY 4.0. Its
headline figure of 4,398 images in version 2 is a base of roughly 1,418 images expanded
three times over by brightness and exposure augmentation ("Outputs per training example:
3"), split 88% train / 8% valid / 4% test. Augmented copies of one photograph can land on
both sides of that split, which inflates any accuracy measured on it. Usable as extra
training data if you deduplicate first and keep the test set strictly from the Kaggle
source. Not usable as a benchmark.

**Kaggle — `youssefmohmmed/dogs-skin-diseases-image-dataset`.** A broader taxonomy
(dermatitis, demodicosis, ringworm, fungal, hypersensitivity, healthy) and larger. Cited
alongside the Motiani set in a 2025 Springer chapter on canine dermatological
classification. Its per-class counts and licence are not documented on its page, so
`configs/datasets/dogs_skin_broad.yaml` ships with a warning and the class list must be
corrected against `--inspect-only` output before training.

**Mendeley — *Lumpy Skin Images Dataset*,
[w36hpf86j2](https://data.mendeley.com/datasets/w36hpf86j2/1).** A real 161 MB zip of
images that downloads without any account, well cited in the cattle-disease literature.
Binary only (lumpy vs healthy cattle), which makes it a weak fit for a project about
classifying *which* skin disease.

## Getting the data

```bash
pip install kaggle
# Kaggle → Settings → Create New Token → save kaggle.json to ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

python scripts/download_dataset.py
python scripts/prepare_data.py
```

Manual download also works. Put the extracted class folders anywhere under
`data/raw/dogs_skin_4class/`; discovery walks the tree recursively, so an extra wrapper
directory from the zip does no harm.

## How images are matched to classes

Folder names in public datasets rarely match configured class keys exactly, so matching
is done on a normalised form: lowercased, with everything non-alphanumeric stripped.
`Bacterial_dermatosis`, `bacterial dermatosis` and `BacterialDermatosis` all resolve to
`bacterial_dermatosis`. Each class also carries an `aliases` list for the names that
differ by more than punctuation, such as `pyoderma` or `ringworm`.

Each image's class is resolved from the nearest ancestor directory that matches, searching
outwards from the file. So `.../Healthy/augmented/x.jpg` is Healthy, and a stray outer
folder sharing a class name cannot override a closer, more specific one. Images in folders
matching nothing are counted and reported rather than dropped silently — the
`--inspect-only` output names them so you can add the alias or delete the folder.

## Splitting, and the leakage trap

443 images is small enough that how you split matters more than usual.

Two things are done about it. Splits are stratified, so each class keeps its proportion in
train, validation and test. And splitting is group-aware when filenames carry an animal
identifier: several photographs of the same dog are kept together in one split. Without
that, near-duplicate shots of one animal land in both train and test, and the reported
accuracy measures memorisation.

The source dataset's filenames determine whether grouping is available. Set
`split.group_pattern` in the dataset config to a regex whose first capture group is the
animal id; `prepare_data.py` then reports how many distinct groups it found, and
`describe_manifest` prints whether any group is shared between splits. For the
Mendeley-style naming scheme (`dog210505_01_02_25`) the pattern is:

```yaml
group_pattern: '^[Dd]og(\d+_\d+_\d+)_\d+$'
```

Leave it `null` if the filenames carry no identifier. Every image then forms its own group
and the split falls back to plain stratified sampling.

## Reporting numbers from this dataset

A single 70/15/15 split puts about 66 images in the test set. Two or three images changing
hands moves accuracy by several points, so one number from one split has an error bar
wide enough to be uninformative.

```bash
python scripts/train.py --crossval
```

That runs stratified (and group-aware) k-fold and reports mean ± standard deviation per
metric, plus per-class recall across folds. Report that.

## Licences

The Kaggle dataset's licence is not stated on its page. The Mendeley label file and the
Roboflow mirror are CC BY 4.0. Check the terms before using any of this beyond coursework
or research, and cite:

- Hwang, S., Shin, H.K., Park, J.M., Kwon, B., Kang, M.-G. (2022). Classification of dog
  skin diseases using deep learning with images captured from multispectral imaging
  device. *Molecular & Cellular Toxicology*, 18. doi:10.1007/s13273-022-00249-7
- Motiani, Y. (2024). Dogs Skin disease dataset. Kaggle.

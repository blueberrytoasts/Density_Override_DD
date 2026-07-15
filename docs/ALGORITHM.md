# Metal Artifact Characterization Pipeline — How It Works

A plain-language walkthrough of the full algorithm, written to support a poster
/ writeup. Every claim is tied to the code (`file:line`) so you can verify it.

**One-sentence summary:** For CT scans of patients with hip implants, we (1)
detect the metal with a self-calibrating star-profile threshold, (2) carve the
remaining tissue into mutually exclusive classes in a fixed "russian doll"
order, and (3) split the ambiguous bright pixels into *bone* vs. *bright
streak artifact* using a multi-feature profile classifier.

---

## The problem

Metal implants are far denser than any tissue, so on CT they (a) saturate
bright and (b) throw **streak artifacts** — bright and dark streaks radiating
from the implant caused by beam hardening and photon starvation. The clinical
problem: bright streak artifacts sit in the **same Hounsfield Unit (HU) range
as real bone** (~300–1500 HU), so a simple brightness threshold cannot tell
"real bone" from "artifact that looks like bone." Getting this split right
matters for radiotherapy dose calculation, where a wrongly-labeled density
changes the planned dose.

The pipeline's job is to label every voxel near the implant as one of:
**metal · dark artifact · bright streak artifact · bone**.

---

## Stage 0 — DICOM loading & HU conversion

`app/dicom_utils.py:load_dicom_series_to_hu()`

- Read the CT series, sort slices by `ImagePositionPatient[2]` so the volume is
  in true anatomical order.
- Convert raw stored pixels to Hounsfield Units with the per-slice rescale:
  `HU = raw × RescaleSlope + RescaleIntercept`. HU is the physical density
  scale everything downstream depends on (air ≈ −1000, water = 0, dense bone
  ≈ +1000, metal ≫ +2000).
- Keep voxel **spacing** (mm) so distances and peak widths can be reported in mm
  rather than pixels.

---

## Stage 1 — Metal detection (self-calibrating, star profiles)

`app/core/metal_detection.py:_detect_adaptive_3d()`

### 1a. Rough guess
Metal is extraordinarily bright, so a first pass is trivial: take the 99.5th
intensity percentile, floored at 2500 HU, and call everything above it
"provisionally metal" (`metal_detection.py:82-85`). This over/under-shoots the
true boundary because a single fixed number can't fit every implant material
(titanium vs. steel glow differently) or every slice.

### 1b. Star-profile refinement (the "FW%" step)
For **each slice**, and **each separate metal blob** on that slice (connected-
component labelling handles bilateral implants — `metal_detection.py:122-131`):

1. Plant a star at the blob's centroid.
2. Shoot **16 radial rays** outward (`compute_star_endpoints`).
3. Walk each ray recording HU. Metal shows up as a tall plateau that falls off
   into tissue — a "mountain" profile.
4. Keep only rays that actually hit metal: peak HU > **75% of the slice's max
   HU** (adaptive filter, not a hardcoded number — generalizes across metals)
   (`metal_detection.py:494-497, 529`).
5. For each valid ray, the candidate threshold is **peak × FW%** (FW% default
   75% → "Full Width at 75% Maximum") (`metal_detection.py:531`).
6. The slice's threshold is the **minimum** across all its rays/blobs — the most
   inclusive choice, so no implant's dim edge is cut off (`metal_detection.py:174-176`).
7. Re-threshold the slice at that value → the refined metal mask.

**Intuition for FW%:** the profile is a mountain. Cutting at 75% of peak height
slices it at a level that hugs the true metal edge, and it re-derives itself per
slice/implant. **Lower FW% → lower cut → fatter metal mask; higher FW% → tighter
mask.** This is the single knob exposed in the UI, and it trades pixels between
the *metal* and *bright artifact* labels (see "How the knobs interact").

### 1c. Cleanup & regions
- **Hole filling:** implants can image as hollow; holes inside the metal are
  filled (`_fill_metal_holes`).
- **Per-implant ROI boxes:** one bounding region per connected metal component
  plus a small margin (`_build_individual_regions`). Critical for **bilateral**
  implants — a single averaged box would span the gap between both hips and drag
  in unrelated tissue. The union of these boxes is `roi_mask`, and it constrains
  all later stages.

---

## Stage 2 — Russian doll segmentation (mutual exclusion)

"Russian doll" = a fixed carve-out **order** where each class only claims pixels
the earlier classes left behind, guaranteeing every voxel gets exactly one label
and no two masks overlap. In the star-profile path
(`pyside_app/segment_worker.py:117-147`, mirrored in `app/main.py:933-954`):

1. **Body mask + ROI** — restrict everything to inside the patient and inside
   the per-implant ROI (`constraint = body_mask & roi_mask`).
2. **Metal** — already claimed in Stage 1.
3. **Dark artifacts** — HU in the dark range (default −1024…−150), *minus metal*.
4. **Bright candidates** — HU in the bright range (default 200…2500), *minus
   metal, minus dark*. These are the pixels that are either real bone or bright
   streak artifact — the ambiguous ones.
5. **Discriminator (Stage 3)** splits the bright candidates into **bone** vs.
   **bright artifact**.

> **The two HU ranges do different jobs.** The **bright range** is a true gate —
> it decides *who becomes a candidate*. The **bone range** (default 400…1800) is
> **not** a gate in this path; it is handed to the discriminator as the "bone
> band" it votes with (Stage 3). The two ranges deliberately overlap. (In the
> *legacy* pure-threshold worker, `segment_worker.py:28-33`, both ranges are
> instead stacked hard cuts with no discriminator — that is the simpler
> "bone = its range, artifact = the rest" behavior.)

---

## Stage 3 — Bone vs. bright-artifact discrimination (star profiles again)

`app/core/discrimination.py:_discriminate_star_profile()`

This is the scientific core: two pixels can have identical HU yet one is bone
and one is artifact. They differ in **local shape**, which we read from radial
profiles.

### 3a. Per-implant stars
For each slice, build **one star per metal implant** (`_get_slice_stars`).
Fragments of a single implant (cup + screws + satellite bits) that lie within
~10 px are merged into one star; bilateral implants stay separate. Each bright
candidate pixel is then judged using the **star of its nearest implant** — so
angle and distance are always measured from the implant the pixel actually
belongs to, not from a midline average between two hips.

### 3b. Four-feature scoring (a weighted panel of judges)
For each bright pixel, we look at its matching ray's profile at that pixel's
distance and compute four features, each casting a weighted vote toward "bone"
(`_analyze_profile_characteristics`, `discrimination.py:737-784`):

| Feature | Bone looks like | Artifact looks like | Weight |
|---|---|---|---|
| **HU value** | inside bone band [400,1800], near its center | far outside the band | **±0.45** |
| **Peak width** | broad, 3–5 mm | narrow, < 2 mm | ±0.35 |
| **Smoothness** | smooth transition (>0.7) | sharp edge (<0.3) | ±0.25 |
| **Gradient** | gentle slope (<50 HU) | steep cliff (>150 HU) | ±0.25 |

The votes sum to a single `bone_score` in roughly [−1.3, +1.3] (range = ±sum of
the four weights). **Decision: `bone_score > 0` → bone, else → bright artifact**
(`_classify_from_profile`, `discrimination.py:805`). A normalized `confidence` in
[0,1] is also stored (normalized by the total weight so it stays in range as
weights change).

**The four weights are tunable** from the PySide app's second toolbar (along
with the bright and bone HU ranges), so the relative influence of HU vs. shape
can be explored per patient. All of them apply on the next *Segment Artifacts*
run — no need to re-detect metal. Defaults reproduce the original behavior.

**Why HU is a vote, not a veto:** streak artifacts frequently land squarely in
the bone HU range, so brightness alone is unreliable. Giving HU the largest
*weight* but still letting the three shape features overrule it is what lets the
method separate bone from same-brightness artifact — the entire point of the
project. (A pixel inside the bone band gets up to +0.45, but a narrow, sharp,
steep profile contributes up to −0.85 and flips it to artifact.)

### 3c. Artifact sub-typing (optional)
Pixels called *artifact* are further tagged as overlying **bone** vs **soft
tissue** using a small HU-neighborhood vote, with distance-from-metal breaking
ties (`_analyze_neighborhood_context`). This is a refinement, not part of the
core bone/artifact call.

---

## How the knobs interact (for the methods section)

- **FW% (metal detection):** only takes effect on **Detect Metal**. Lower FW% →
  bigger metal mask → the innermost bright ring is labeled *metal* and removed
  from the candidate pool → less bright-artifact. Higher FW% → that ring is left
  for the discriminator → more bright-artifact. Total (metal + bright artifact)
  stays roughly constant; FW% just sets where the metal/artifact boundary sits.
  **Pick one value and keep it constant across all patients/figures.**
- **Bright range:** the hard gate for what gets analyzed at all.
- **Bone range:** the discriminator's HU band — a strong vote (weight `w_hu`,
  default 0.45), not a cut, in the star-profile path.
- **Feature weights (`w_hu / w_width / w_smooth / w_gradient`):** how loudly each
  of the four votes counts. Raising `w_hu` makes the classifier trust HU more
  (behaves more like a threshold); raising the shape weights lets geometry
  override HU more often. Bright/bone ranges and all four weights are exposed on
  the PySide "Discrimination" toolbar and apply on the next Segment run.
- **Star angles (16/32):** angular resolution of the profiles; more angles =
  finer directional sampling, slower.

---

## Color legend (matches the app overlays)

| Color | Label |
|---|---|
| Red | Metal implant |
| Yellow | Bright streak artifact |
| Magenta | Dark artifact |
| Blue | Bone |
| Lime | ROI boundary |

---

## Pipeline at a glance

```
DICOM → HU volume
     │
     ▼
[1] METAL DETECTION
     rough 99.5th-pct/≥2500 HU threshold
       → per-slice, per-implant 16-ray star profiles
       → threshold = peak × FW% (min across rays)  ← FW% knob
       → hole-fill, per-implant ROI boxes (bilateral-safe)  → metal mask, roi_mask
     │
     ▼
[2] RUSSIAN DOLL (mutual exclusion, inside body ∩ ROI)
     metal ▶ dark artifacts (dark range − metal)
           ▶ bright candidates (bright range − metal − dark)   ← "who gets judged"
     │
     ▼
[3] DISCRIMINATION  (per bright candidate)
     one star per implant → nearest-ray profile
       → 4 weighted features: HU ±0.45, peak-width ±0.35, smoothness ±0.25, gradient ±0.25
       → bone_score > 0 ? BONE (blue) : BRIGHT ARTIFACT (yellow)
```

---

## Reproducing / visualizing for figures

`tools/visualize_discriminator.py "data/HIP3 Patient" [--slice N] [--angles 16]`

Renders the exact discriminator geometry for one slice: the CT with each
implant's star and rays, the resulting bone/artifact labels, and a per-ray plot
of the HU profile with every judged pixel drawn at its (distance, HU) colored by
verdict. Good source material for a "how the classifier sees the data" figure.
Defaults match the app thresholds; run picks the slice with the most metal.

---

*Key source files:* `app/dicom_utils.py`, `app/core/metal_detection.py`,
`app/core/discrimination.py`, `app/contour_operations.py`,
`pyside_app/segment_worker.py`, `app/main.py`.

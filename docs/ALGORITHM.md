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

### 1c. Cleanup
- **Hole filling:** implants can image as hollow; holes inside the metal are
  filled (`_fill_metal_holes`).

### 1d. How the ROI (region of interest) is created
The ROI is the neighborhood around the implant(s) that all later stages are
restricted to — it stops the segmentation from wandering off into unrelated
anatomy. It is built **per implant**, not as one big box, which is what makes
bilateral cases work. Steps (`_build_individual_regions`, `_regions_to_roi_mask`):

1. **Find implants per slice.** On each slice with metal, run connected-component
   labelling (4-connectivity) so the left and right implants are separate objects
   (`metal_detection.py:367-369`). Specks < 4 px are dropped as noise.
2. **One box per implant.** For each component, take its bounding box and pad it
   by a **1 cm margin** (converted mm→voxels using the in-plane spacing), and
   record the component's centroid (`metal_detection.py:386-394`). The margin
   leaves room for artifact that extends just beyond the metal.
3. **Merge only if genuinely overlapping.** If two boxes on the same slice
   overlap by more than 20 % of the smaller box's area, they're fused into their
   union, repeated until nothing else merges (`_merge_overlapping_boxes`). Two
   *separate* hips don't overlap, so they stay in their own boxes — but a single
   implant that fragmented into several components gets stitched back together.
4. **Union into a 3-D mask.** All per-implant boxes across all slices are OR-ed
   into a single 3-D boolean `roi_mask`, with an **extra 5-voxel "conservative"
   margin** so artifact just outside a box is still captured
   (`_regions_to_roi_mask`).

**Why per-implant and not one box:** a single box around *all* metal in a
bilateral patient would span the gap between the two hips and pull in the bladder,
bowel, and midline soft tissue. The per-implant `roi_mask` hugs each hip
separately and leaves the gap out. This `roi_mask` — not the overall
`roi_bounds` extent, which is kept only for display — is what constrains Stages 2
and 3 (`constraint = body_mask & roi_mask`). It can be shown in the app as the
lime **ROI** overlay.

---

## Stage 2 — Russian doll segmentation (mutual exclusion)

"Russian doll" = a fixed carve-out **order** where each class only claims pixels
the earlier classes left behind, guaranteeing every voxel gets exactly one label
and no two masks overlap. In the star-profile path
(`pyside_app/segment_worker.py:117-147`, mirrored in `app/main.py:933-954`):

1. **Body mask + ROI** — restrict everything to inside the patient and inside
   the per-implant ROI (`constraint = body_mask & roi_mask`).
2. **Metal** — already claimed in Stage 1.
3. **Dark artifacts** — HU in the dark range (default −1024…−150, tunable on
   the toolbar), *minus metal*.
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

### 3b. Four-feature scoring (weighted vote)
For each bright pixel, we look at its matching ray's profile at that pixel's
distance and compute four features, each casting a weighted vote toward "bone"
(`_analyze_profile_characteristics`, `discrimination.py:737-784`):

| Feature (app label) | The question it votes on | Bone looks like | Artifact looks like | Weight |
|---|---|---|---|---|
| **HU** | Is the pixel's brightness inside the bone HU band? | inside band [400,1800], near its center | far outside the band | **±0.45** |
| **width** | Is the intensity peak broad or a thin streak? | broad, 3–5 mm | narrow, < 2 mm | ±0.35 |
| **smooth** | Does brightness fade gradually or stop at a sharp edge? | smooth transition (>0.7) | sharp edge (<0.3) | ±0.25 |
| **grad** | Is the slope into surrounding tissue gentle or a cliff? | gentle slope (<50 HU/mm) | steep cliff (>150 HU/mm) | ±0.25 |

Each feature casts a **vote**: positive (looks like bone) or negative (looks like
artifact), scaled by its **weight** — how loudly that clue counts. A weight of 0
mutes that clue. The four labels above (**HU / width / smooth / grad**) are
exactly the four weight boxes on the PySide "Discrimination" toolbar.

#### How each feature is actually measured

All three *shape* features are read off the same 1-D **HU profile** — the string
of HU values sampled along the pixel's matching ray (`_get_star_profiles_detailed`),
one sample per pixel step, so samples sit **~1 mm apart** (0.98 mm in-plane on the
HIP data). That raw profile is first **Gaussian-smoothed** (σ = 2 samples, ~2 mm)
to suppress noise, and its **gradient** (point-to-point change, `np.gradient`) is
computed. To judge one pixel, the code jumps to the profile sample nearest that
pixel's distance from the star centre and takes a small **local window** of ±5
samples (~11 points, **≈ 10 mm** of profile centred on the pixel) of the smoothed
HU and its gradient (`_analyze_profile_characteristics`,
`discrimination.py:737-744`). All three shape features — width, smoothness, and
gradient — are computed on that same 10 mm window:

- **Peak width** (`_calculate_peak_width`, mm) — Full-Width-at-Half-Maximum: find
  the max HU in the window, count how many samples sit above **half** of that
  max, and multiply by the in-plane pixel spacing. A broad plateau clears
  half-max over many samples → wide (bone); a thin spike clears it over one or
  two → narrow (artifact). Votes **+** if > 3 mm, **−** if < 2 mm.
- **Smoothness** (`_calculate_smoothness_score`, 0–1) — take the **variance of the
  gradient** across the window, then `smoothness = 1 / (1 + variance/100)`. If HU
  changes steadily (small gradient variance) the score is near 1 (smooth = bone);
  if the gradient jumps around, it drops toward 0 (rough edge = artifact). Votes
  **+** if > 0.7, **−** if < 0.3 — which corresponds to gradient variance below
  **~43** (bone) and above **~233** (artifact). The 0–1 score is the quantity the
  code actually compares; the variance figures are just those cutoffs inverted.
- **Gradient magnitude** (HU per mm) — the absolute value of the **mean gradient**
  across the window: how steep the HU slope is right at the pixel. Since samples
  are ~1 mm apart, `np.gradient` is already ~HU/mm. Small = gentle ramp (bone);
  large = steep wall (artifact edge). Votes **+** if < 50, **−** if > 150.

The **HU** feature needs no profile — it just compares the pixel's own HU to the
bone band, scoring most positive at the band's centre and turning negative once
the pixel is outside it (fully negative ~500 HU beyond the edge). Note the
neutral zones: e.g. a peak width between 2 and 3 mm casts no width vote at all,
so only clearly bone-like or clearly artifact-like values move the score.

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

### 3c. Artifact sub-typing (over bone vs. over tissue)
Once a pixel is called *artifact*, a second question matters for density
override: what tissue is the artifact corrupting *underneath* (so it can be
restored to the right HU)? This decision does **not** look at the artifact
pixel itself — it takes a majority vote of the pixel's HU **neighborhood**:
count neighbors in the **context bone band** vs. the **context tissue band**,
and assign the pixel to whichever dominates (distance-from-metal breaks ties —
close to metal → tissue, i.e. muscle around the implant; far → bone). Both
artifact classes are split this way:

- **Bright artifacts** — the star discriminator emits this alongside its
  bone/artifact verdict (`artifact_bone_mask` / `artifact_tissue_mask`,
  `discrimination.py:_discriminate_star_profile` → `_analyze_neighborhood_context`).
- **Dark artifacts** — split by `classify_artifacts_contextually`
  (`contour_operations.py`, banded 5-3-1 score). This is a per-voxel loop, so in
  the PySide app it runs **lazily** (`ContextSplitWorker`) the first time a
  dark-split overlay is viewed, rather than on every Segment; the result is
  cached until the next segmentation.
- **Legacy method** — the legacy threshold worker computes no splits itself;
  both the bright and dark splits run lazily through the same
  `ContextSplitWorker` path when their overlays are first viewed. Note this
  means legacy's *bright* split uses the banded 5-3-1 score, not the
  discriminator's simple majority vote (the two agree except in mixed
  neighborhoods).

The neighborhood size is the **Context window** knob (in-plane px, ±1 slice in
z), shared by both splits (default 5). The old fixed sizes were 5×5×3 (bright)
and 7×7×3 (dark); they are now unified and tunable.

**The context bands are their own knobs, separate from Decision 1's bone vote
band.** Decision 1's "Bone HU" band decides *what HU is bone-like for the vote*;
the context bands decide *what surrounding HU counts as bone vs. tissue when
judging what's underneath*. They default to **500–1500** (context bone) and
**−100–300** (context tissue), and both the bright and dark splits share them,
so the two paths stay consistent. In the star-profile path they are threaded as
`ctx_bone_low/high` and `tissue_hu_low/high`; `ctx_bone_low/high` default to the
vote band when a caller omits them (preserving legacy behavior), but the PySide
worker always passes the toolbar values.

Each split partitions its parent mask exactly (every parent voxel lands in
bone *or* tissue, never both). Both are surfaced in the PySide app as opt-in
overlays (orange/pink for bright, blue-violet/teal for dark) and work with
both segmentation methods; the star-profile worker returns the bright split
eagerly, everything else is computed lazily on first view. This is a
refinement, not part of the core bone/artifact call.

---

## How the knobs interact (for the methods section)

- **FW% (metal detection):** only takes effect on **Detect Metal**. Lower FW% →
  bigger metal mask → the innermost bright ring is labeled *metal* and removed
  from the candidate pool → less bright-artifact. Higher FW% → that ring is left
  for the discriminator → more bright-artifact. Total (metal + bright artifact)
  stays roughly constant; FW% just sets where the metal/artifact boundary sits.
  **Pick one value and keep it constant across all patients/figures.**
- **Dark range:** plain HU threshold for dark artifacts (default −1024…−150).
  Raising the upper bound toward 0 grabs more of the gray streak shadows (and
  eventually normal fat, ~−100 HU). Applies to both segmentation methods.
- **Bright range:** the hard gate for what gets analyzed at all.
- **Bone range:** the discriminator's HU band — a strong vote (weight `w_hu`,
  default 0.45), not a cut, in the star-profile path. This is **Decision 1**
  (bone vs. artifact).
- **Context bone / context tissue ranges:** **Decision 2** knobs — which
  neighboring HU counts as bone vs. soft tissue when deciding what an artifact
  overlies (over-bone vs. over-tissue split). Defaults 500–1500 / −100–300;
  drive both the bright and dark splits. Independent of the Decision-1 bone
  vote band above.
- **Context window (px):** in-plane size of the neighborhood the Decision-2
  vote counts over (default 5, i.e. 5×5 in-plane, ±1 slice). Bigger = an
  artifact pixel tallies bone/tissue farther away, so a pixel bordering bone
  leans bone even if its immediate neighbors are tissue — the lever when a
  pink region sits *next to* bone rather than *over* bone-HU. Drives both
  splits; too large and the vote blurs across tissue boundaries.
- **Feature weights (`w_hu / w_width / w_smooth / w_gradient`):** how loudly each
  of the four votes counts. Raising `w_hu` makes the classifier trust HU more
  (behaves more like a threshold); raising the shape weights lets geometry
  override HU more often. Dark/bright/bone ranges and all four weights are
  exposed on the PySide "Discrimination" toolbar and apply on the next Segment
  run.
- **Star angles (16/32):** angular resolution of the profiles; more angles =
  finer directional sampling, slower.

### Fixing a specific misclassification (practical recipe)

Two different "this should be bone" complaints have two different fixes — read
the overlay color to tell which:

- **A pink pixel (bright→tissue) that should be over bone.** This is a
  **Decision 2** error. Lower the **Context bone HU** floor (e.g. 500 → ~350–400)
  and re-Segment. Peri-implant bone, blurred by beam hardening and partial
  volume, often reads below 500 HU and falls into the 300–500 "dead zone" that
  counts as neither bone nor tissue — so the neighborhood vote defaults to
  tissue. Lowering the floor lets that bone register as bone *neighbors*.
  Because the label is a **local majority vote of the neighborhood**, not a test
  of the pixel itself, this flips only the regions genuinely surrounded by bone
  (e.g. one side of an implant) and leaves artifact-over-muscle pink. Nudge in
  small steps: if pink that really is over muscle starts flipping, the floor
  has dropped into soft-tissue HU and gone too far. Optionally also lower the
  **Context tissue HU** ceiling (300 → ~150) so mid-density pixels stop counting
  as tissue votes.
- **A yellow/pink pixel that should be real *bone*, not an artifact at all.**
  This is a **Decision 1** error (the bone-vs-artifact call itself). Widen the
  **Bone HU** vote band or raise `w_hu`, or lower the shape weights so geometry
  overrides HU less. This moves pixels from the artifact masks into the blue
  bone mask.

Not tunable: an artifact pixel within 1.5 cm of metal whose neighborhood vote is
an exact bone/tissue tie is hardcoded to fall to tissue. It rarely matters, but
right at the implant edge it can hold a thin rim pink.

---

## Color legend (matches the app overlays)

| Color | Label | What it is |
|---|---|---|
| Red | Metal | Detected metal implant (Stage 1) |
| Magenta | Dark artifact | Dark streak / photon-starvation regions |
| Yellow | Bright artifact | Bright streak artifact (Stage 3 verdict) |
| Blue | Bone | Real bone (Stage 3 verdict) |
| Orange | Bright→bone | Bright artifact corrupting bone underneath (Stage 4) |
| Hot pink | Bright→tissue | Bright artifact corrupting soft tissue underneath (Stage 4) |
| Blue-violet | Dark→bone | Dark artifact corrupting bone underneath (Stage 4) |
| Teal | Dark→tissue | Dark artifact corrupting soft tissue underneath (Stage 4) |
| Lime | ROI | Per-implant ROI box outline (Stage 1d) |
| Cyan | Metal stars | The FW% rays shot from each metal blob to set the metal threshold (Stage 1b) |
| White | Disc stars | The discriminator's rays shot from each implant centroid to judge bone vs. artifact (Stage 3a) |

Each overlay is an independent toggle in the PySide legend, so figures can show
any combination (e.g. Metal + Disc stars to illustrate the classifier geometry).
"Metal stars" appears after **Detect Metal**; "Disc stars" appears after
**Segment Artifacts** (Star Profile method). Both, and the ROI, are diagnostic
overlays that are off by default.

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

**In the app (quickest):** toggle the **Disc stars** (white) overlay after
running Segment Artifacts to see, on any slice you scrub to, exactly where the
discriminator planted its stars and shot its rays — one star per implant. Pair it
with **Metal** (red) for a clean classifier-geometry figure, and use **Export
Slice…** for a 4× lossless PNG. Toggle **Metal stars** (cyan) instead to show the
Stage-1 FW% detection rays.

**Export Legend…** writes a standalone color-key PNG (also 4×) for placing next
to a figure. It includes exactly the tissue/artifact classes currently toggled
on, so the key matches your exported slice; the diagnostic overlays (ROI, metal
stars, disc stars) are never included. Toggle the classes you want shown, then
export.

**Standalone diagnostic plot:**
`tools/visualize_discriminator.py "data/HIP3 Patient" [--slice N] [--angles 16]`

Renders the exact discriminator geometry for one slice *plus* the per-ray HU
profiles: the CT with each implant's star and rays, the resulting bone/artifact
labels, and a plot per ray with every judged pixel drawn at its (distance, HU)
colored by verdict. Good source material for a "how the classifier sees the data"
figure. Defaults match the app thresholds; run picks the slice with the most
metal.

---

*Key source files:* `app/dicom_utils.py`, `app/core/metal_detection.py`,
`app/core/discrimination.py`, `app/contour_operations.py`,
`pyside_app/segment_worker.py`, `app/main.py`.

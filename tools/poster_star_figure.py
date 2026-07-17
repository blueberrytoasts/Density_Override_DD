"""
Poster figure: Decision 1 (star-profile discrimination) for one slice.

Simplified variant of visualize_discriminator.py for print: all rays are drawn
in a single muted color, TWO rays are highlighted, and only those two rays'
HU-vs-distance profiles are shown beside the slice — one bone-dominated ray
(broad, smooth plateau) and one artifact-dominated ray (narrow, steep spike).

Usage (from repo root):
  python tools/poster_star_figure.py "data/HIP4 Patient" --slice 184
      [--angles 16] [--rays BONE_RAY,ARTIFACT_RAY]
      [--bone-low 400] [--bone-high 1800]
      [--bright-low 200] [--bright-high 2500] [--out output/poster_fig.png]

If --rays is omitted the script picks the ray with the most bone-classified
voxels and the ray with the most artifact-classified voxels.
Defaults match the PySide app's star-profile thresholds (segment_worker.py).
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "app"))

from dicom_utils import load_dicom_series_to_hu
from body_mask import create_body_mask
from core.metal_detection import MetalDetector, MetalDetectionMethod
from core.discrimination import ArtifactDiscriminator, DiscriminationMethod

# Highlight colors: distinct from the class colors (red metal, blue bone,
# yellow artifact) and from each other.
BONE_RAY_COLOR = "#00e5ff"      # cyan  — the bone-example ray
ARTIFACT_RAY_COLOR = "#ff00d0"  # magenta — the artifact-example ray
DIM_RAY_COLOR = (1, 1, 1, 0.35)  # all other rays


def find_ct_dir(patient_dir):
    """Return the subdirectory with the most .dcm files (the CT series)."""
    best, best_count = patient_dir, 0
    for root, _dirs, files in os.walk(patient_dir):
        count = sum(1 for f in files if f.lower().endswith(".dcm"))
        if count > best_count:
            best, best_count = root, count
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patient", help="Patient folder (e.g. 'data/HIP4 Patient')")
    parser.add_argument("--slice", type=int, default=None,
                        help="Slice index (default: slice with most metal)")
    parser.add_argument("--angles", type=int, default=16)
    parser.add_argument("--rays", default=None,
                        help="Two ray indices 'bone_ray,artifact_ray' (default: auto-pick)")
    parser.add_argument("--bone-low", type=float, default=400.0)
    parser.add_argument("--bone-high", type=float, default=1800.0)
    parser.add_argument("--bright-low", type=float, default=200.0)
    parser.add_argument("--bright-high", type=float, default=2500.0)
    parser.add_argument("--dark-high", type=float, default=-150.0)
    parser.add_argument("--ymax", type=float, default=2600.0,
                        help="Top of the profile y-axis (raise to ~5400 to show metal peaks)")
    parser.add_argument("--no-dots", action="store_true",
                        help="Draw the profile lines only, without the judged-pixel dots")
    parser.add_argument("--out", default=None, help="Output PNG path")
    args = parser.parse_args()

    ct_dir = find_ct_dir(args.patient)
    print(f"Loading CT from: {ct_dir}")
    ct_volume, meta = load_dicom_series_to_hu(ct_dir)
    if ct_volume is None:
        sys.exit("No CT data found.")
    spacing = np.abs(meta["spacing"])
    print(f"Volume {ct_volume.shape}, spacing {spacing}")

    print("Running metal detection (full volume)...")
    detector = MetalDetector(MetalDetectionMethod.ADAPTIVE_3D)
    detection = detector.detect(ct_volume, tuple(spacing))
    metal_mask = detection["mask"]
    roi_mask = detection.get("roi_mask")
    if not np.any(metal_mask):
        sys.exit("No metal detected.")

    metal_per_slice = metal_mask.sum(axis=(1, 2))
    z = args.slice if args.slice is not None else int(np.argmax(metal_per_slice))
    if not metal_per_slice[z]:
        sys.exit(f"Slice {z} contains no metal; metal spans slices "
                 f"{np.flatnonzero(metal_per_slice)[0]}-{np.flatnonzero(metal_per_slice)[-1]}")
    print(f"Using slice {z} ({metal_per_slice[z]} metal pixels)")

    # Candidate masks — same construction as the PySide star-profile worker
    body_mask = create_body_mask(ct_volume, air_threshold=-300)
    constraint = body_mask & roi_mask if roi_mask is not None else body_mask
    dark_mask = (ct_volume >= -1024) & (ct_volume <= args.dark_high) & ~metal_mask & constraint
    bright_mask = ((ct_volume >= args.bright_low) & (ct_volume <= args.bright_high)
                   & ~metal_mask & ~dark_mask & constraint)
    print(f"Bright candidates on slice {z}: {bright_mask[z].sum()}")

    disc = ArtifactDiscriminator(DiscriminationMethod.STAR_PROFILE)
    result = disc.discriminate(
        ct_volume[z:z + 1], metal_mask[z:z + 1], bright_mask[z:z + 1],
        tuple(spacing), num_angles=args.angles,
        bone_hu_low=args.bone_low, bone_hu_high=args.bone_high, use_gpu=False,
    )
    bone_px = result["bone_mask"][0]
    artifact_px = result["artifact_mask"][0]
    print(f"Classified: bone={bone_px.sum()}, artifact={artifact_px.sum()}")

    stars = disc._get_slice_stars(ct_volume[z], metal_mask[z], args.angles)
    star_centers = np.array([[s["center_y"], s["center_x"]] for s in stars])
    print(f"Stars on this slice: {len(stars)}")

    # Assign judged pixels to (star, ray) exactly as the discriminator does
    ray_points = {(s, i): {"bone": [], "artifact": []}
                  for s in range(len(stars)) for i in range(args.angles)}
    px_spacing = float(np.mean(spacing[:2]))
    for label, mask2d in (("bone", bone_px), ("artifact", artifact_px)):
        ys, xs = np.where(mask2d)
        for py, px in zip(ys, xs):
            d2 = (star_centers[:, 0] - py) ** 2 + (star_centers[:, 1] - px) ** 2
            s_idx = int(np.argmin(d2))
            cy, cx = star_centers[s_idx]
            angle = np.arctan2(py - cy, px - cx)
            if angle < 0:
                angle += 2 * np.pi
            idx = int((angle / (2 * np.pi)) * args.angles) % args.angles
            dist_mm = np.hypot(py - cy, px - cx) * px_spacing
            ray_points[(s_idx, idx)][label].append((dist_mm, ct_volume[z, py, px]))

    # Pick the two rays to highlight (on star 0)
    if args.rays:
        bone_ray, artifact_ray = (int(v) for v in args.rays.split(","))
    else:
        bone_counts = [len(ray_points[(0, i)]["bone"]) for i in range(args.angles)]
        art_counts = [len(ray_points[(0, i)]["artifact"]) for i in range(args.angles)]
        # Bone ray: most bone dots weighted by bone fraction, so a ray that is
        # mostly bone beats one that is merely busy. Artifact ray: same, inverted.
        totals = [b + a for b, a in zip(bone_counts, art_counts)]
        bone_ray = int(np.argmax([b * (b / t) if t else 0
                                  for b, t in zip(bone_counts, totals)]))
        artifact_ray = int(np.argmax([a * (a / t) if t else 0
                                      for a, t in zip(art_counts, totals)]))
        if artifact_ray == bone_ray:
            artifact_ray = int(np.argsort(art_counts)[-2])
    print(f"Highlighting rays: bone-example {bone_ray}, artifact-example {artifact_ray}")

    # ---- Figure 1: the slice with rays (standalone) ----
    fig_slice, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(ct_volume[z], cmap="gray", vmin=-200, vmax=1800)
    for mask2d, color in ((metal_mask[z], (1, 0, 0, 0.85)),
                          (bone_px, (0.1, 0.35, 1, 0.65)),
                          (artifact_px, (1, 1, 0, 0.65))):
        overlay = np.zeros((*mask2d.shape, 4))
        overlay[mask2d] = color
        ax.imshow(overlay)

    star = stars[0]
    cy, cx = star["center_y"], star["center_x"]
    for i, p in enumerate(star["profiles"]):
        r = min(p["distances"].max(), 130)
        ey, ex = cy + r * np.sin(p["angle"]), cx + r * np.cos(p["angle"])
        if i == bone_ray:
            color, lw, zorder = BONE_RAY_COLOR, 3.0, 5
        elif i == artifact_ray:
            color, lw, zorder = ARTIFACT_RAY_COLOR, 3.0, 5
        else:
            color, lw, zorder = DIM_RAY_COLOR, 0.6, 4
        ax.plot([cx, ex], [cy, ey], color=color, lw=lw, zorder=zorder)
    # Remaining stars (bilateral cases): dim rays only
    for s2 in stars[1:]:
        cy2, cx2 = s2["center_y"], s2["center_x"]
        for p in s2["profiles"]:
            r = min(p["distances"].max(), 130)
            ax.plot([cx2, cx2 + r * np.cos(p["angle"])],
                    [cy2, cy2 + r * np.sin(p["angle"])],
                    color=DIM_RAY_COLOR, lw=0.6, zorder=4)
    ax.axis("off")

    # ---- Figure 2: the two highlighted profiles (standalone, stacked) ----
    fig_prof, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    fig_prof.subplots_adjust(hspace=0.3)
    for row, (ray_idx, color, label) in enumerate((
            (bone_ray, BONE_RAY_COLOR, "broad, smooth plateau → bone"),
            (artifact_ray, ARTIFACT_RAY_COLOR, "narrow, steep spike → artifact"))):
        p = star["profiles"][ray_idx]
        axp = axes[row]
        d_mm = p["distances"] * px_spacing
        axp.axhspan(args.bone_low, args.bone_high, color="tab:blue", alpha=0.10)
        axp.plot(d_mm, p["hu_values"], color=color, lw=2.2)
        if not args.no_dots:
            for cls, dot_color in (("bone", "tab:blue"), ("artifact", "gold")):
                pts = ray_points[(0, ray_idx)][cls]
                if pts:
                    pd, phu = zip(*pts)
                    axp.scatter(pd, phu, s=16, color=dot_color, zorder=3,
                                edgecolors="black", linewidths=0.3)
        axp.set_xlim(0, 130)
        axp.set_ylim(-300, args.ymax)
        axp.set_ylabel("HU", fontsize=11)
        axp.tick_params(labelsize=10)
        if row == 1:
            axp.set_xlabel("Distance from implant center (mm)", fontsize=11)
        axp.text(0.98, 0.9, "bone HU band", transform=axp.transAxes,
                 fontsize=9, color="tab:blue", alpha=0.8, ha="right")

    base = args.out or os.path.join(
        REPO_ROOT, "output",
        f"poster_star_{os.path.basename(os.path.normpath(args.patient)).replace(' ', '_')}"
        f"_slice{z}.png")
    os.makedirs(os.path.dirname(base), exist_ok=True)
    stem, ext = os.path.splitext(base)
    out_slice = f"{stem}_slice{ext}" if not stem.endswith(f"slice{z}") else f"{stem}_image{ext}"
    out_prof = f"{stem}_profiles{ext}"
    fig_slice.savefig(out_slice, dpi=300, bbox_inches="tight", facecolor="black")
    fig_prof.savefig(out_prof, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_slice}")
    print(f"Saved: {out_prof}")


if __name__ == "__main__":
    main()

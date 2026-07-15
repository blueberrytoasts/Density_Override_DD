"""
Visualize the discriminator's star profiles for one slice.

The star-profile discriminator shoots one 16/32-ray star per connected metal
component on each slice (discrimination.py:_get_slice_stars) — bilateral
implants each get their own star. Each bright pixel is assigned to its nearest
component's star, matched to the nearest ray by angle, and judged using that
ray's profile shape at the pixel's distance. None of this is drawn in the
apps — this script reproduces the exact geometry and shows:

  Left panel : CT slice + metal (red) + the discriminator's rays (colored by
               angle) + resulting classification (bone=blue, artifact=yellow)
  Right grid : each ray's HU-vs-distance profile in its matching color, with
               the bone HU band shaded and every bright pixel judged by that
               ray plotted at (distance, HU) — blue dot = called bone,
               yellow dot = called artifact.

Usage (from repo root):
  python tools/visualize_discriminator.py "data/HIP3 Patient" [--slice N]
      [--angles 16] [--bone-low 150] [--bone-high 1500]
      [--bright-low 800] [--bright-high 3500] [--out output/disc_debug.png]

Defaults match the Streamlit app's russian_doll thresholds (main.py:839-846).
Slice defaults to the slice with the most metal.
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
    parser.add_argument("patient", help="Patient folder (e.g. 'data/HIP3 Patient')")
    parser.add_argument("--slice", type=int, default=None,
                        help="Slice index (default: slice with most metal)")
    parser.add_argument("--angles", type=int, default=16,
                        help="Number of rays (app slider default is 32; 16 is easier to read)")
    parser.add_argument("--bone-low", type=float, default=150.0)
    parser.add_argument("--bone-high", type=float, default=1500.0)
    parser.add_argument("--bright-low", type=float, default=800.0)
    parser.add_argument("--bright-high", type=float, default=3500.0)
    parser.add_argument("--dark-high", type=float, default=-150.0)
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

    # Pick slice: most metal pixels
    metal_per_slice = metal_mask.sum(axis=(1, 2))
    z = args.slice if args.slice is not None else int(np.argmax(metal_per_slice))
    if not metal_per_slice[z]:
        sys.exit(f"Slice {z} contains no metal; metal spans slices "
                 f"{np.flatnonzero(metal_per_slice)[0]}-{np.flatnonzero(metal_per_slice)[-1]}")
    print(f"Using slice {z} ({metal_per_slice[z]} metal pixels)")

    # Build candidate masks exactly like the Streamlit star-profile path (main.py:908-941)
    body_mask = create_body_mask(ct_volume, air_threshold=-300)
    constraint = body_mask & roi_mask if roi_mask is not None else body_mask
    dark_mask = (ct_volume >= -1024) & (ct_volume <= args.dark_high) & ~metal_mask & constraint
    bright_mask = ((ct_volume >= args.bright_low) & (ct_volume <= args.bright_high)
                   & ~metal_mask & ~dark_mask & constraint)
    print(f"Bright candidates on slice {z}: {bright_mask[z].sum()}")

    # Run the discriminator on this slice only (identical result for this slice:
    # everything it computes is per-slice except the distance transform, which
    # only breaks ties for artifact sub-typing, not the bone/artifact call)
    disc = ArtifactDiscriminator(DiscriminationMethod.STAR_PROFILE)
    result = disc.discriminate(
        ct_volume[z:z + 1], metal_mask[z:z + 1], bright_mask[z:z + 1],
        tuple(spacing), num_angles=args.angles,
        bone_hu_low=args.bone_low, bone_hu_high=args.bone_high, use_gpu=False,
    )
    bone_px = result["bone_mask"][0]
    artifact_px = result["artifact_mask"][0]
    print(f"Classified: bone={bone_px.sum()}, artifact={artifact_px.sum()}")

    # Reproduce the stars exactly as the discriminator builds them
    # (one per connected metal component — bilateral-safe)
    stars = disc._get_slice_stars(ct_volume[z], metal_mask[z], args.angles)
    star_centers = np.array([[s["center_y"], s["center_x"]] for s in stars])
    print(f"Stars on this slice: {len(stars)} at " +
          ", ".join(f"(y={s['center_y']}, x={s['center_x']})" for s in stars))

    # Assign each judged pixel to its star + ray (same rules as the discriminator)
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

    # ---- Figure ----
    ncols_grid = 4
    n_plots = len(stars) * args.angles
    nrows_grid = int(np.ceil(n_plots / ncols_grid))
    fig = plt.figure(figsize=(22, max(11, 2.4 * nrows_grid)))
    gs = GridSpec(nrows_grid, ncols_grid + 4, figure=fig, hspace=0.6, wspace=0.35)
    ray_colors = plt.cm.hsv(np.linspace(0, 1, args.angles, endpoint=False))
    star_markers = "sox^D*"  # centroid marker per star on the left panel

    # Left: slice + rays + classification
    ax = fig.add_subplot(gs[:, :4])
    ax.imshow(ct_volume[z], cmap="gray", vmin=-200, vmax=1800)
    for name, mask2d, color in (("metal", metal_mask[z], (1, 0, 0, 0.85)),
                                ("bone", bone_px, (0.1, 0.35, 1, 0.65)),
                                ("artifact", artifact_px, (1, 1, 0, 0.65))):
        overlay = np.zeros((*mask2d.shape, 4))
        overlay[mask2d] = color
        ax.imshow(overlay)
    max_r = max(p["distances"].max() for s in stars for p in s["profiles"])
    for s_idx, star in enumerate(stars):
        cy, cx = star["center_y"], star["center_x"]
        for i, p in enumerate(star["profiles"]):
            r = min(p["distances"].max(), 130)
            ey, ex = cy + r * np.sin(p["angle"]), cx + r * np.cos(p["angle"])
            ax.plot([cx, ex], [cy, ey], color=ray_colors[i], lw=1.0, alpha=0.85)
        ax.plot(cx, cy, "w" + star_markers[s_idx % len(star_markers)], ms=10, mew=2,
                mfc="none")
        ax.text(cx, cy - 12, f"S{s_idx}", color="white", fontsize=11,
                fontweight="bold", ha="center")
    ax.set_title(f"Slice {z}: {len(stars)} discriminator star(s) "
                 f"({args.angles} rays each, one per metal component)\n"
                 f"red=metal  blue=classified bone  yellow=classified artifact", fontsize=11)
    ax.axis("off")

    # Right: one subplot per (star, ray)
    for s_idx, star in enumerate(stars):
        for i, p in enumerate(star["profiles"]):
            k = s_idx * args.angles + i
            axp = fig.add_subplot(gs[k // ncols_grid, 4 + k % ncols_grid])
            d_mm = p["distances"] * px_spacing
            axp.axhspan(args.bone_low, args.bone_high, color="tab:blue", alpha=0.10)
            axp.plot(d_mm, p["hu_values"], color=ray_colors[i], lw=1.0)
            for label, color in (("bone", "tab:blue"), ("artifact", "gold")):
                pts = ray_points[(s_idx, i)][label]
                if pts:
                    pd, phu = zip(*pts)
                    axp.scatter(pd, phu, s=6, color=color, zorder=3,
                                edgecolors="black", linewidths=0.2)
            axp.set_title(f"S{s_idx} ray {i} ({np.degrees(p['angle']):.0f}°)  "
                          f"b:{len(ray_points[(s_idx, i)]['bone'])} "
                          f"a:{len(ray_points[(s_idx, i)]['artifact'])}",
                          fontsize=8, color=ray_colors[i] * 0.8)
            axp.set_xlim(0, min(max_r * px_spacing, 180))
            axp.set_ylim(-300, 2600)
            axp.tick_params(labelsize=7)
            if k // ncols_grid == nrows_grid - 1:
                axp.set_xlabel("mm from center", fontsize=8)
            if k % ncols_grid == 0:
                axp.set_ylabel("HU", fontsize=8)

    fig.suptitle(
        f"{os.path.basename(os.path.normpath(args.patient))} — discriminator star profiles — "
        f"bone HU [{args.bone_low:.0f},{args.bone_high:.0f}] shaded; "
        f"dots = judged pixels (blue→bone, yellow→artifact)", fontsize=13)

    out = args.out or os.path.join(
        REPO_ROOT, "output",
        f"discriminator_debug_{os.path.basename(os.path.normpath(args.patient)).replace(' ', '_')}_slice{z}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

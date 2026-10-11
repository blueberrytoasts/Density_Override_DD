"""Compare two RTSTRUCTs drawn on the same CT, structure by structure.

Typical use: app contours (exported from the PySide app) vs physician-approved
contours exported from MIM, both on the same CT series.

Usage:
    # See what structures each file contains
    python tools/compare_contours.py --ct "data/HIP3 Patient" \\
        --test RS_app.dcm --ref RS_mim.dcm --list

    # Compare; structures with the same name (case-insensitive) pair up
    # automatically, others are paired with --match TEST=REF
    python tools/compare_contours.py --ct "data/HIP3 Patient" \\
        --test RS_app.dcm --ref RS_mim.dcm \\
        --match "Metal=Implant" --match "Bright_Over_Bone=BrightArt_Bone" \\
        --out output/compare_hip3.csv

``--ct`` may be the CT series folder or a patient folder containing it.
Metrics: Dice, Jaccard, volumes (cc), volume difference, HD95 / max HD / mean
surface distance (mm), sensitivity (fraction of ref covered) and precision
(fraction of test inside ref). Both RTSTRUCTs are rasterized the same way onto
the CT grid, so differences reflect the contours, not the rasterization.

Output contains structure names and metrics only — no patient identifiers.
"""
import argparse
import csv
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "algorithms"))

from dicom_utils import load_dicom_series_to_hu, resolve_ct_dir  # noqa: E402
from rtstruct import (CTGeometry, read_rtstruct, rasterize_structure,  # noqa: E402
                      referenced_series_uid)
from contour_metrics import compare_masks  # noqa: E402

COLUMNS = ('test', 'ref', 'dice', 'jaccard', 'test_cc', 'ref_cc',
           'volume_diff_cc', 'volume_diff_pct', 'hd95_mm', 'hd_max_mm',
           'msd_mm', 'sensitivity', 'precision')


def pair_structures(test_names, ref_names, explicit):
    """Explicit TEST=REF pairs first, then case-insensitive name matches."""
    pairs, used_t, used_r = [], set(), set()
    for spec in explicit:
        if '=' not in spec:
            raise SystemExit(f"--match must be TEST=REF, got: {spec}")
        t, r = (x.strip() for x in spec.split('=', 1))
        if t not in test_names:
            raise SystemExit(f"--match: '{t}' not in test RTSTRUCT (try --list)")
        if r not in ref_names:
            raise SystemExit(f"--match: '{r}' not in reference RTSTRUCT (try --list)")
        pairs.append((t, r))
        used_t.add(t)
        used_r.add(r)
    ref_by_lower = {n.lower(): n for n in ref_names if n not in used_r}
    for t in test_names:
        r = ref_by_lower.get(t.lower())
        if t not in used_t and r is not None:
            pairs.append((t, r))
            used_t.add(t)
            used_r.add(r)
    return pairs


def _fmt(v):
    if isinstance(v, str):
        return v
    return '   nan' if v != v else f"{v:8.3f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--ct', required=True, help='CT series folder (or patient folder)')
    ap.add_argument('--test', required=True, help='RTSTRUCT to evaluate (e.g. app export)')
    ap.add_argument('--ref', required=True, help='Reference RTSTRUCT (e.g. physician contours)')
    ap.add_argument('--match', action='append', default=[], metavar='TEST=REF',
                    help='Pair structures with different names (repeatable)')
    ap.add_argument('--list', action='store_true', help='List structure names and exit')
    ap.add_argument('--out', help='Write results to this CSV file')
    args = ap.parse_args(argv)

    test = {s['name']: s for s in read_rtstruct(args.test)}
    ref = {s['name']: s for s in read_rtstruct(args.ref)}

    if args.list:
        for label, structs in (('test', test), ('ref', ref)):
            print(f"{label}:")
            for name, s in structs.items():
                print(f"  {name:<32} {len(s['contours']):5d} contours")
        return 0

    ct_dir = resolve_ct_dir(args.ct)
    # Load the CT series the test RTSTRUCT was drawn on, so another series
    # in the same folder can't be picked by mistake.
    series_uid = referenced_series_uid(args.test)
    volume, meta = load_dicom_series_to_hu(ct_dir, series_uid) if ct_dir else (None, None)
    if volume is None and series_uid is not None:
        print("WARNING: the CT series referenced by --test is not in --ct; "
              "falling back to the largest CT series there.", file=sys.stderr)
        volume, meta = load_dicom_series_to_hu(ct_dir)
    if volume is None:
        print("Could not load a CT series from --ct", file=sys.stderr)
        return 1
    geom = CTGeometry.from_meta(meta, volume.shape)
    spacing = geom.voxel_spacing_zyx()

    for label, structs in (('test', test), ('ref', ref)):
        frames = {s['frame_of_reference_uid'] for s in structs.values()} - {''}
        if frames and frames != {meta['frame_of_reference_uid']}:
            print(f"WARNING: {label} RTSTRUCT references a different Frame of "
                  f"Reference than the CT — contours may not align.", file=sys.stderr)

    pairs = pair_structures(list(test), list(ref), args.match)
    if not pairs:
        print("No structures paired. Use --list and --match TEST=REF.", file=sys.stderr)
        return 1

    rows = []
    for t_name, r_name in pairs:
        t_mask, t_off = rasterize_structure(test[t_name], geom)
        r_mask, r_off = rasterize_structure(ref[r_name], geom)
        for label, name, off in (('test', t_name, t_off), ('ref', r_name, r_off)):
            if off:
                print(f"WARNING: {off} {label} contour(s) of '{name}' are not on a "
                      f"CT slice and were skipped.", file=sys.stderr)
        row = {'test': t_name, 'ref': r_name}
        row.update(compare_masks(t_mask, r_mask, spacing))
        rows.append(row)

    print(f"{'test':<22} {'ref':<22} {'dice':>8} {'hd95_mm':>8} {'msd_mm':>8} "
          f"{'test_cc':>8} {'ref_cc':>8}")
    for row in rows:
        print(f"{row['test'][:22]:<22} {row['ref'][:22]:<22} "
              + " ".join(_fmt(row[k]) for k in ('dice', 'hd95_mm', 'msd_mm', 'test_cc', 'ref_cc')))

    unpaired_t = sorted(set(test) - {t for t, _ in pairs})
    unpaired_r = sorted(set(ref) - {r for _, r in pairs})
    if unpaired_t or unpaired_r:
        print(f"\nUnpaired — test: {unpaired_t or '-'}; ref: {unpaired_r or '-'}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows({k: row[k] for k in COLUMNS} for row in rows)
        print(f"\nWrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

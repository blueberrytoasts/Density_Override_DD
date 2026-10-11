"""RTSTRUCT export/import round trip on synthetic CT series (no patient data).

Builds fake CT DICOM series with an off-centre origin, unequal row/column
spacing and several patient orientations, then checks that:
  - the loader keeps the geometry and UIDs,
  - exported contours sit at the right patient coordinates,
  - export -> read -> rasterize reproduces the original masks,
  - the RTSTRUCT references the CT's Frame of Reference, series and slices,
  - the comparison metrics behave on known shapes,
  - tools/compare_contours.py runs end to end.

Run:  python tests/test_rtstruct_roundtrip.py   (or: pytest tests/)
"""
import os
import sys
import tempfile

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'algorithms'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'tools'))

from dicom_utils import load_dicom_series_to_hu  # noqa: E402
from rtstruct import (CTGeometry, create_rtstruct, read_rtstruct,  # noqa: E402
                      rasterize_structure, save_rtstruct)
from contour_metrics import compare_masks  # noqa: E402
import compare_contours  # noqa: E402

SHAPE = (12, 48, 64)                 # slices, rows, cols
ROW_SP, COL_SP, SLICE_SP = 0.8, 1.1, 2.5
ORIGIN = np.array([-123.4, 56.7, -300.0])
ORIENTATIONS = {
    'HFS': [1, 0, 0, 0, 1, 0],
    'HFP': [-1, 0, 0, 0, -1, 0],
    'FFS': [-1, 0, 0, 0, 1, 0],
}


def write_fake_ct(folder, iop):
    """Write a synthetic CT series; returns (frame_of_ref, series_uid, study_uid)."""
    frame_uid, series_uid, study_uid = generate_uid(), generate_uid(), generate_uid()
    row_cos, col_cos = np.array(iop[:3], float), np.array(iop[3:], float)
    normal = np.cross(row_cos, col_cos)
    for k in range(SHAPE[0]):
        fm = FileMetaDataset()
        fm.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
        fm.MediaStorageSOPInstanceUID = generate_uid()
        fm.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(None, {}, file_meta=fm, preamble=b'\0' * 128)
        ds.SOPClassUID = fm.MediaStorageSOPClassUID
        ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
        ds.PatientName, ds.PatientID = 'Phantom^Synthetic', 'PHANTOM'
        ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, series_uid
        ds.FrameOfReferenceUID = frame_uid
        ds.Modality = 'CT'
        ds.ImageOrientationPatient = [str(v) for v in iop]
        ds.ImagePositionPatient = [f"{v:.3f}" for v in ORIGIN + k * SLICE_SP * normal]
        ds.PixelSpacing = [str(ROW_SP), str(COL_SP)]
        ds.SliceThickness = str(SLICE_SP)
        ds.Rows, ds.Columns = SHAPE[1], SHAPE[2]
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, 'MONOCHROME2'
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.RescaleSlope, ds.RescaleIntercept = 1, 0
        ds.PixelData = np.zeros(SHAPE[1:], dtype=np.int16).tobytes()
        ds.save_as(os.path.join(folder, f'CT{k:03d}.dcm'), enforce_file_format=True)
    return frame_uid, series_uid, study_uid


def make_masks():
    z, r, c = np.indices(SHAPE)
    blob = ((z - 6) / 4.0) ** 2 + ((r - 20) / 10.0) ** 2 + ((c - 22) / 12.0) ** 2 <= 1.0
    # Ring with a hole on every slice it occupies
    rr = np.hypot(r - 30, c - 48)
    ring = (rr >= 4) & (rr <= 9) & (z >= 3) & (z <= 8)
    # Touches the image edge (row 0 and last column)
    edge = (r <= 5) & (c >= 56) & (z >= 1) & (z <= 4)
    return {'Blob': blob, 'Ring': ring, 'Edge': edge}


def _export_roundtrip(iop):
    with tempfile.TemporaryDirectory() as tmp:
        ct_dir = os.path.join(tmp, 'CT')
        os.makedirs(ct_dir)
        frame_uid, series_uid, _ = write_fake_ct(ct_dir, iop)
        volume, meta = load_dicom_series_to_hu(ct_dir)
        masks = make_masks()
        structures = [(n, m, (255, 0, 0)) for n, m in masks.items()]
        ds, exported = create_rtstruct(structures, meta, volume.shape)
        path = os.path.join(tmp, 'RS.dcm')
        save_rtstruct(ds, path)
        back = {s['name']: s for s in read_rtstruct(path)}
        geom = CTGeometry.from_meta(meta, volume.shape)
        return (volume, meta, masks, exported, pydicom.dcmread(path), back, geom,
                frame_uid, series_uid)


def test_loader_keeps_geometry_and_uids():
    with tempfile.TemporaryDirectory() as tmp:
        frame_uid, series_uid, _ = write_fake_ct(tmp, ORIENTATIONS['HFS'])
        volume, meta = load_dicom_series_to_hu(tmp)
    assert volume.shape == SHAPE
    assert meta['frame_of_reference_uid'] == frame_uid
    assert meta['series_instance_uid'] == series_uid
    assert len(meta['sop_instance_uids']) == SHAPE[0]
    assert meta['pixel_spacing_rc'] == (ROW_SP, COL_SP)
    np.testing.assert_allclose(meta['spacing'], [COL_SP, ROW_SP, SLICE_SP])
    np.testing.assert_allclose(meta['image_position_patient'][0], ORIGIN, atol=1e-3)


def test_roundtrip_reproduces_masks_all_orientations():
    for name, iop in ORIENTATIONS.items():
        _, _, masks, exported, _, back, geom, _, _ = _export_roundtrip(iop)
        assert sorted(exported) == sorted(masks), name
        for sname, original in masks.items():
            restored, off = rasterize_structure(back[sname], geom)
            assert off == 0, (name, sname)
            mismatch = int((restored ^ original).sum())
            assert mismatch == 0, f"{name}/{sname}: {mismatch} voxels differ"


# Hand-written (not derived from the code under test): sign of patient x per
# +1 column, sign of patient y per +1 row, sign of patient z per +1 slice
# written by write_fake_ct.
EXPECTED_SIGNS = {'HFS': (1, 1, 1), 'HFP': (-1, -1, 1), 'FFS': (-1, 1, -1)}


def test_contour_points_at_expected_patient_coordinates():
    for name, iop in ORIENTATIONS.items():
        sx, sy, sz = EXPECTED_SIGNS[name]
        _, _, masks, _, _, back, _, _, _ = _export_roundtrip(iop)
        blob = masks['Blob']
        pts = np.concatenate(back['Blob']['contours'])
        # The outline sits half a voxel outside the outermost voxel centres.
        cols = np.nonzero(blob.any(axis=(0, 1)))[0]
        rows = np.nonzero(blob.any(axis=(0, 2)))[0]
        x_edges = sorted(ORIGIN[0] + sx * np.array([cols.min() - 0.5, cols.max() + 0.5]) * COL_SP)
        y_edges = sorted(ORIGIN[1] + sy * np.array([rows.min() - 0.5, rows.max() + 0.5]) * ROW_SP)
        assert np.allclose([pts[:, 0].min(), pts[:, 0].max()], x_edges, atol=1e-3), name
        assert np.allclose([pts[:, 1].min(), pts[:, 1].max()], y_edges, atol=1e-3), name
        # Volume slices are sorted by ascending z, whatever the write order.
        slice_z = np.sort(ORIGIN[2] + sz * np.arange(SHAPE[0]) * SLICE_SP)
        zs = np.nonzero(blob.any(axis=(1, 2)))[0]
        assert np.allclose([pts[:, 2].min(), pts[:, 2].max()],
                           [slice_z[zs.min()], slice_z[zs.max()]], atol=1e-3), name


def test_rtstruct_references_ct():
    _, meta, _, _, ds, _, _, frame_uid, series_uid = _export_roundtrip(ORIENTATIONS['HFS'])
    assert ds.Modality == 'RTSTRUCT'
    assert ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.3'
    ref_frame = ds.ReferencedFrameOfReferenceSequence[0]
    assert ref_frame.FrameOfReferenceUID == frame_uid
    ref_series = ref_frame.RTReferencedStudySequence[0].RTReferencedSeriesSequence[0]
    assert ref_series.SeriesInstanceUID == series_uid
    assert [i.ReferencedSOPInstanceUID for i in ref_series.ContourImageSequence] == \
        meta['sop_instance_uids']
    assert ds.PatientID == 'PHANTOM'  # copied from CT so it files under the same patient
    assert ds.StudyInstanceUID == meta['patient_study']['StudyInstanceUID']
    for roi_contour in ds.ROIContourSequence:
        for item in roi_contour.ContourSequence:
            uid = item.ContourImageSequence[0].ReferencedSOPInstanceUID
            z = meta['sop_instance_uids'].index(uid)
            pts = np.asarray(item.ContourData, float).reshape(-1, 3)
            assert np.allclose(pts[:, 2], meta['image_position_patient'][z][2])


def test_loader_picks_one_series_from_mixed_folder():
    """Two CT series in one folder must not be merged."""
    with tempfile.TemporaryDirectory() as tmp:
        a_dir, b_dir, mixed = (os.path.join(tmp, d) for d in ('a', 'b', 'mixed'))
        for d in (a_dir, b_dir, mixed):
            os.makedirs(d)
        _, series_a, _ = write_fake_ct(a_dir, ORIENTATIONS['HFS'])
        _, series_b, _ = write_fake_ct(b_dir, ORIENTATIONS['HFS'])
        for f in sorted(os.listdir(a_dir)):
            os.rename(os.path.join(a_dir, f), os.path.join(mixed, 'A_' + f))
        # Series B: same positions (e.g. a second kernel) but only 5 slices
        for f in sorted(os.listdir(b_dir))[:5]:
            os.rename(os.path.join(b_dir, f), os.path.join(mixed, 'B_' + f))

        volume, meta = load_dicom_series_to_hu(mixed)
        assert volume.shape == SHAPE
        assert meta['series_instance_uid'] == series_a
        assert meta['other_series_ignored'] == 1
        assert meta['duplicate_slices_dropped'] == 0
        assert CTGeometry.from_meta(meta, volume.shape).voxel_spacing_zyx()[0] == SLICE_SP

        volume_b, meta_b = load_dicom_series_to_hu(mixed, series_b)
        assert volume_b.shape == (5,) + SHAPE[1:]
        assert meta_b['series_instance_uid'] == series_b
        assert load_dicom_series_to_hu(mixed, '1.2.3.404') == (None, None)


def test_long_contour_roundtrip():
    """A contour with thousands of points (> 64 KB of ContourData) must survive."""
    shape = (3, 512, 512)
    z, r, c = np.indices(shape)
    ang = np.arctan2(r - 256, c - 256)
    # Star-shaped outline: very jagged, so the traced contour is long.
    radius = 150 + 60 * np.sign(np.sin(ang * 90))
    mask = (np.hypot(r - 256, c - 256) <= radius) & (z == 1)
    meta = {
        'image_position_patient': np.array([[0.0, 0.0, k * 2.0] for k in range(3)]),
        'image_orientation_patient': np.array([1, 0, 0, 0, 1, 0], float),
        'pixel_spacing_rc': (0.9, 0.9),
        'sop_instance_uids': [generate_uid() for _ in range(3)],
        'sop_class_uid': '1.2.840.10008.5.1.4.1.1.2',
        'series_instance_uid': generate_uid(),
        'frame_of_reference_uid': generate_uid(),
        'patient_study': {},
    }
    ds, _ = create_rtstruct([('Jagged', mask, (1, 2, 3))], meta, shape)
    longest = max(int(i.NumberOfContourPoints)
                  for i in ds.ROIContourSequence[0].ContourSequence)
    assert longest > 3000, longest
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'RS.dcm')
        save_rtstruct(ds, path)
        back = read_rtstruct(path)[0]
    restored, off = rasterize_structure(back, CTGeometry.from_meta(meta, shape))
    assert off == 0 and not (restored ^ mask).any()


def test_metrics_on_known_shapes():
    spacing = (SLICE_SP, ROW_SP, COL_SP)
    a = np.zeros(SHAPE, bool)
    a[3:9, 10:30, 10:30] = True
    same = compare_masks(a, a, spacing)
    assert same['dice'] == 1.0 and same['hd95_mm'] == 0.0 and same['msd_mm'] == 0.0
    shifted = np.roll(a, 2, axis=2)  # 2 columns
    res = compare_masks(shifted, a, spacing)
    assert abs(res['hd_max_mm'] - 2 * COL_SP) < 1e-6
    assert abs(res['dice'] - 18 / 20) < 1e-9
    empty = compare_masks(np.zeros(SHAPE, bool), a, spacing)
    assert empty['dice'] == 0.0 and np.isnan(empty['hd95_mm'])


def test_compare_contours_cli():
    with tempfile.TemporaryDirectory() as tmp:
        ct_dir = os.path.join(tmp, 'CT')
        os.makedirs(ct_dir)
        write_fake_ct(ct_dir, ORIENTATIONS['HFS'])
        volume, meta = load_dicom_series_to_hu(ct_dir)
        masks = make_masks()
        ds, _ = create_rtstruct([(n, m, (0, 0, 255)) for n, m in masks.items()],
                                meta, volume.shape)
        save_rtstruct(ds, os.path.join(tmp, 'test.dcm'))
        # Reference: same blob under another name, ring shifted one column
        ref = {'BLOB_MD': masks['Blob'], 'Ring': np.roll(masks['Ring'], 1, axis=2)}
        ds, _ = create_rtstruct([(n, m, (0, 255, 0)) for n, m in ref.items()],
                                meta, volume.shape)
        save_rtstruct(ds, os.path.join(tmp, 'ref.dcm'))
        out = os.path.join(tmp, 'res.csv')
        rc = compare_contours.main(['--ct', tmp, '--test', os.path.join(tmp, 'test.dcm'),
                                    '--ref', os.path.join(tmp, 'ref.dcm'),
                                    '--match', 'Blob=BLOB_MD', '--out', out])
        assert rc == 0
        import csv
        rows = {r['test']: r for r in csv.DictReader(open(out))}
        assert float(rows['Blob']['dice']) == 1.0
        assert 0.7 < float(rows['Ring']['dice']) < 1.0


if __name__ == '__main__':
    tests = [v for k, v in dict(globals()).items() if k.startswith('test_')]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}")
    print(f"\nAll {len(tests)} RTSTRUCT round-trip tests passed.")

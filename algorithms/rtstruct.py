"""DICOM RT Structure Set (RTSTRUCT) export and import.

Both directions share one geometry model (``CTGeometry``) built from the CT
metadata returned by ``load_dicom_series_to_hu``, so a mask exported here and
read back lands on exactly the same voxels.

Coordinates follow the DICOM image plane equation (PS3.3 C.7.6.2.1.1):

    P(row, col) = IPP_slice + col * col_spacing * X + row * row_spacing * Y

where X / Y are the row / column direction cosines from
ImageOrientationPatient, and IPP_slice is that slice's ImagePositionPatient
(the centre of its first voxel). Mask index (z, row, col) is volume order.
"""
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, PYDICOM_IMPLEMENTATION_UID, generate_uid
from skimage import measure
from skimage.draw import polygon as draw_polygon

RT_STRUCTURE_SET_STORAGE = '1.2.840.10008.5.1.4.1.1.481.3'
DETACHED_STUDY_MANAGEMENT = '1.2.840.10008.3.1.2.3.1'


@dataclass
class CTGeometry:
    """Voxel <-> patient-coordinate mapping for a loaded CT volume."""
    ipp: np.ndarray            # (n_slices, 3) ImagePositionPatient per slice
    row_cos: np.ndarray        # (3,) direction of increasing column index
    col_cos: np.ndarray        # (3,) direction of increasing row index
    row_spacing: float         # mm between rows (PixelSpacing[0])
    col_spacing: float         # mm between columns (PixelSpacing[1])
    shape: tuple               # (n_slices, rows, cols)

    @classmethod
    def from_meta(cls, meta: dict, shape) -> "CTGeometry":
        iop = np.asarray(meta['image_orientation_patient'], dtype=float)
        row_sp, col_sp = meta['pixel_spacing_rc']
        return cls(
            ipp=np.asarray(meta['image_position_patient'], dtype=float),
            row_cos=iop[:3], col_cos=iop[3:6],
            row_spacing=float(row_sp), col_spacing=float(col_sp),
            shape=tuple(shape),
        )

    @property
    def normal(self) -> np.ndarray:
        n = np.cross(self.row_cos, self.col_cos)
        return n / np.linalg.norm(n)

    @property
    def slice_offsets(self) -> np.ndarray:
        """Each slice's position along the plane normal (mm)."""
        return self.ipp @ self.normal

    def voxel_spacing_zyx(self) -> tuple:
        """(slice, row, col) spacing in mm, for distance computations."""
        offsets = self.slice_offsets
        dz = float(np.median(np.abs(np.diff(offsets)))) if len(offsets) > 1 else 1.0
        return (dz, self.row_spacing, self.col_spacing)

    def to_patient(self, z: int, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Fractional (row, col) indices on slice ``z`` -> (N, 3) patient mm."""
        return (self.ipp[z][None, :]
                + np.outer(cols * self.col_spacing, self.row_cos)
                + np.outer(rows * self.row_spacing, self.col_cos))

    def to_index(self, points: np.ndarray, tol_mm: float | None = None):
        """(N, 3) coplanar patient points -> (slice index, rows, cols).

        Returns ``(None, None, None)`` if the plane doesn't match a slice
        within ``tol_mm`` (default: half the slice spacing).
        """
        offsets = self.slice_offsets
        plane = float(np.mean(points @ self.normal))
        z = int(np.argmin(np.abs(offsets - plane)))
        if tol_mm is None:
            tol_mm = 0.5 * self.voxel_spacing_zyx()[0]
        if abs(offsets[z] - plane) > tol_mm:
            return None, None, None
        rel = points - self.ipp[z][None, :]
        cols = (rel @ self.row_cos) / self.col_spacing
        rows = (rel @ self.col_cos) / self.row_spacing
        return z, rows, cols


# ---- export -----------------------------------------------------------------
def _ds_str(value: float) -> str:
    """Format a coordinate as a valid DICOM DS (<= 16 chars)."""
    return f"{value:.3f}"


def _mask_slice_contours(mask_2d: np.ndarray):
    """Closed iso-contours of a 2D mask as (row, col) arrays.

    The mask is zero-padded so shapes touching the image edge still give
    closed contours; the 0.5 level puts the outline on the voxel boundary.
    Holes come back as separate contours, which planning systems treat as
    cut-outs of the enclosing contour.
    """
    padded = np.pad(mask_2d.astype(np.uint8), 1)
    for contour in measure.find_contours(padded, 0.5):
        contour = contour - 1.0  # undo padding
        if len(contour) > 1 and np.allclose(contour[0], contour[-1]):
            contour = contour[:-1]  # DICOM closes implicitly
        if len(contour) >= 3:
            yield contour


def create_rtstruct(structures, meta: dict, shape,
                    label: str = 'ArtifactSeg',
                    series_description: str = 'Metal artifact contours'):
    """Build an RTSTRUCT referencing the CT described by ``meta``.

    Args:
        structures: iterable of ``(name, mask, (r, g, b))``; each mask is a
            3D bool array in volume order. Empty masks are skipped.
        meta: metadata dict from ``load_dicom_series_to_hu``.
        shape: CT volume shape (n_slices, rows, cols).

    Returns:
        (pydicom FileDataset, list of exported structure names)
    """
    geom = CTGeometry.from_meta(meta, shape)
    sop_uids = meta['sop_instance_uids']
    ct_sop_class = meta.get('sop_class_uid') or '1.2.840.10008.5.1.4.1.1.2'
    now = datetime.now()

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = RT_STRUCTURE_SET_STORAGE
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # Patient + General Study modules: copied from the CT so the planning
    # system files this under the same patient and study.
    for kw, value in meta.get('patient_study', {}).items():
        setattr(ds, kw, value)
    for kw in ('PatientName', 'PatientID', 'PatientBirthDate', 'PatientSex',
               'StudyDate', 'StudyTime', 'StudyID', 'AccessionNumber',
               'ReferringPhysicianName'):
        if kw not in ds:
            setattr(ds, kw, '')  # Type 2: must be present, may be empty
    if 'StudyInstanceUID' not in ds:
        ds.StudyInstanceUID = generate_uid()

    # RT Series + General Equipment
    ds.Modality = 'RTSTRUCT'
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = 1
    ds.SeriesDescription = series_description
    ds.OperatorsName = ''
    ds.Manufacturer = 'Density_Override_DD'

    # SOP Common
    ds.SOPClassUID = RT_STRUCTURE_SET_STORAGE
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.InstanceCreationDate = now.strftime('%Y%m%d')
    ds.InstanceCreationTime = now.strftime('%H%M%S')

    # Structure Set module
    ds.StructureSetLabel = label[:16]
    ds.StructureSetDate = now.strftime('%Y%m%d')
    ds.StructureSetTime = now.strftime('%H%M%S')

    def image_ref(uid):
        item = Dataset()
        item.ReferencedSOPClassUID = ct_sop_class
        item.ReferencedSOPInstanceUID = uid
        return item

    series_ref = Dataset()
    series_ref.SeriesInstanceUID = meta['series_instance_uid']
    series_ref.ContourImageSequence = Sequence([image_ref(u) for u in sop_uids])
    study_ref = Dataset()
    study_ref.ReferencedSOPClassUID = DETACHED_STUDY_MANAGEMENT
    study_ref.ReferencedSOPInstanceUID = ds.StudyInstanceUID
    study_ref.RTReferencedSeriesSequence = Sequence([series_ref])
    frame_ref = Dataset()
    frame_ref.FrameOfReferenceUID = meta['frame_of_reference_uid']
    frame_ref.RTReferencedStudySequence = Sequence([study_ref])
    ds.ReferencedFrameOfReferenceSequence = Sequence([frame_ref])

    roi_seq, contour_seq, obs_seq, exported = [], [], [], []
    for name, mask, color in structures:
        if mask is None or not np.any(mask):
            continue
        number = len(roi_seq) + 1

        roi = Dataset()
        roi.ROINumber = number
        roi.ReferencedFrameOfReferenceUID = meta['frame_of_reference_uid']
        roi.ROIName = name[:64]
        roi.ROIGenerationAlgorithm = 'AUTOMATIC'
        roi_seq.append(roi)

        contours = []
        for z in np.flatnonzero(mask.reshape(mask.shape[0], -1).any(axis=1)):
            for rc in _mask_slice_contours(mask[z]):
                pts = geom.to_patient(int(z), rc[:, 0], rc[:, 1])
                item = Dataset()
                item.ContourImageSequence = Sequence([image_ref(sop_uids[z])])
                item.ContourGeometricType = 'CLOSED_PLANAR'
                item.NumberOfContourPoints = len(pts)
                item.ContourData = [_ds_str(v) for v in pts.ravel()]
                contours.append(item)

        roi_contour = Dataset()
        roi_contour.ReferencedROINumber = number
        roi_contour.ROIDisplayColor = [int(c) for c in color]
        roi_contour.ContourSequence = Sequence(contours)
        contour_seq.append(roi_contour)

        obs = Dataset()
        obs.ObservationNumber = number
        obs.ReferencedROINumber = number
        obs.RTROIInterpretedType = 'ORGAN'
        obs.ROIInterpreter = ''
        obs_seq.append(obs)
        exported.append(roi.ROIName)

    ds.StructureSetROISequence = Sequence(roi_seq)
    ds.ROIContourSequence = Sequence(contour_seq)
    ds.RTROIObservationsSequence = Sequence(obs_seq)
    return ds, exported


def save_rtstruct(ds, output_path):
    """Write an RTSTRUCT dataset to disk."""
    ds.save_as(output_path, enforce_file_format=True)


# ---- import -----------------------------------------------------------------
def read_rtstruct(path) -> list[dict]:
    """Read an RTSTRUCT into a list of structures.

    Returns:
        list of ``{'number', 'name', 'color', 'frame_of_reference_uid',
        'contours'}`` where ``contours`` is a list of (N, 3) patient-mm
        arrays (CLOSED_PLANAR only; points/open contours are skipped).
    """
    ds = pydicom.dcmread(path)
    if getattr(ds, 'Modality', '') != 'RTSTRUCT':
        raise ValueError(f"Not an RTSTRUCT: {path}")

    rois = {}
    for roi in getattr(ds, 'StructureSetROISequence', []):
        rois[int(roi.ROINumber)] = {
            'number': int(roi.ROINumber),
            'name': str(roi.ROIName),
            'color': None,
            'frame_of_reference_uid': str(getattr(roi, 'ReferencedFrameOfReferenceUID', '')),
            'contours': [],
        }
    for rc in getattr(ds, 'ROIContourSequence', []):
        entry = rois.get(int(rc.ReferencedROINumber))
        if entry is None:
            continue
        if 'ROIDisplayColor' in rc:
            entry['color'] = tuple(int(c) for c in rc.ROIDisplayColor)
        for item in getattr(rc, 'ContourSequence', []):
            if getattr(item, 'ContourGeometricType', '') != 'CLOSED_PLANAR':
                continue
            pts = np.asarray(item.ContourData, dtype=float).reshape(-1, 3)
            if len(pts) >= 3:
                entry['contours'].append(pts)
    return list(rois.values())


def rasterize_structure(structure: dict, geom: CTGeometry) -> tuple[np.ndarray, int]:
    """Fill a structure's contours onto the CT voxel grid.

    A voxel is inside if its centre is inside the contour. Multiple contours
    on the same slice are combined with XOR, so a contour drawn inside
    another is a hole (how RTSTRUCT encodes holes).

    Returns:
        (3D bool mask in volume order, number of contours that didn't fall
        on any CT slice and were skipped)
    """
    mask = np.zeros(geom.shape, dtype=bool)
    off_grid = 0
    for pts in structure['contours']:
        z, rows, cols = geom.to_index(pts)
        if z is None:
            off_grid += 1
            continue
        rr, cc = draw_polygon(rows, cols, shape=geom.shape[1:])
        layer = np.zeros(geom.shape[1:], dtype=bool)
        layer[rr, cc] = True
        mask[z] ^= layer
    return mask, off_grid

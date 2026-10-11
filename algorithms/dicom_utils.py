import numpy as np
import os
import pydicom

CT_IMAGE_STORAGE = '1.2.840.10008.5.1.4.1.1.2'

# Patient/study attributes copied verbatim into derived objects (e.g. an
# RTSTRUCT) so the planning system files them under the same patient and
# study as the CT. Held in memory only; never logged.
_PATIENT_STUDY_KEYWORDS = (
    'SpecificCharacterSet',
    'PatientName', 'PatientID', 'PatientBirthDate', 'PatientSex',
    'StudyInstanceUID', 'StudyDate', 'StudyTime', 'StudyID',
    'AccessionNumber', 'ReferringPhysicianName',
)


def resolve_ct_dir(root: str) -> str | None:
    """Find the directory that holds the CT slices under ``root``.

    Patient folders nest the CT series in a subfolder alongside an RTSTRUCT
    folder. The CT series has hundreds of files while RTSTRUCT has ~1, so we
    pick the directory (at any depth, including ``root`` itself) that directly
    contains the most files. This is fast (no DICOM parsing) and robust across
    the differing folder-naming conventions in ``data/``.

    Returns:
        Absolute path to the best candidate directory, or ``None`` if ``root``
        contains no files at all.
    """
    best_dir = None
    best_count = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        n = len(filenames)
        if n > best_count:
            best_count = n
            best_dir = dirpath
    return best_dir


def load_dicom_series_to_hu(dicom_dir, series_instance_uid=None):
    """
    Load a DICOM series from a directory and convert to Hounsfield Units.

    Only one series is loaded: the one matching ``series_instance_uid`` if
    given, otherwise the CT series with the most slices. Other CT series in
    the folder (a second reconstruction kernel, a scout) are ignored rather
    than merged in, and slices repeating an already-seen position are dropped.

    Args:
        dicom_dir: Path to directory containing DICOM files
        series_instance_uid: Load this series specifically (e.g. the one an
            RTSTRUCT references). Returns (None, None) if it isn't there.

    Returns:
        tuple: (3D numpy array of HU values indexed [slice, row, col],
                dict with spatial metadata)

    Metadata keys:
        spacing: (x, y, z) voxel spacing in mm — column spacing, row spacing,
            slice spacing. This is the order every algorithm expects.
        origin, shape, slice_z_positions, slice_thickness: as before.
        image_position_patient: (n_slices, 3) ImagePositionPatient per slice,
            in volume order.
        image_orientation_patient: (6,) row then column direction cosines.
        pixel_spacing_rc: (row spacing, column spacing) in mm.
        sop_instance_uids, sop_class_uid, series_instance_uid,
        frame_of_reference_uid: identifiers needed to reference this CT
            from an RTSTRUCT.
        patient_study: dict of patient/study attributes to copy into
            derived objects (see _PATIENT_STUDY_KEYWORDS).
        other_series_ignored: number of other CT series found in the folder.
        duplicate_slices_dropped: slices skipped for repeating a position.
    """
    slices = []
    for s in os.listdir(dicom_dir):
        try:
            filepath = os.path.join(dicom_dir, s)
            ds = pydicom.dcmread(filepath)
            if ds.SOPClassUID == CT_IMAGE_STORAGE and 'PixelData' in ds:
                slices.append(ds)
        except (pydicom.errors.InvalidDicomError, AttributeError, KeyError):
            # Skip non-DICOM files or files with missing required attributes
            continue

    if not slices:
        print(f"No valid CT DICOM images found in {dicom_dir}")
        return None, None

    by_series = {}
    for ds in slices:
        by_series.setdefault(str(getattr(ds, 'SeriesInstanceUID', '')), []).append(ds)
    if series_instance_uid is not None:
        if series_instance_uid not in by_series:
            print(f"Requested CT series not found in {dicom_dir}")
            return None, None
        chosen = series_instance_uid
    else:
        chosen = max(by_series, key=lambda uid: len(by_series[uid]))
    other_series_ignored = len(by_series) - 1
    slices = by_series[chosen]

    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    unique, seen = [], set()
    for ds in slices:
        pos = tuple(round(float(v), 3) for v in ds.ImagePositionPatient)
        if pos not in seen:
            seen.add(pos)
            unique.append(ds)
    duplicate_slices_dropped = len(slices) - len(unique)
    slices = unique

    pixel_spacing = slices[0].PixelSpacing
    slice_thickness = float(slices[0].SliceThickness)
    spacing_between_slices = getattr(slices[0], 'SpacingBetweenSlices', slice_thickness)
    image_position_patient = np.array(slices[0].ImagePositionPatient)
    origin_3d = image_position_patient
    spacing_3d = np.array([pixel_spacing[1], pixel_spacing[0], spacing_between_slices])

    image_3d_shape = (len(slices), slices[0].Rows, slices[0].Columns)
    image_3d_hu = np.zeros(image_3d_shape, dtype=np.int16)

    for i, s in enumerate(slices):
        raw_pixel_array = s.pixel_array
        rescale_slope = getattr(s, 'RescaleSlope', 1)
        rescale_intercept = getattr(s, 'RescaleIntercept', 0)
        hu_array = raw_pixel_array * rescale_slope + rescale_intercept
        image_3d_hu[i, :, :] = hu_array

    slice_z_positions = np.array([float(s.ImagePositionPatient[2]) for s in slices])

    first = slices[0]
    return image_3d_hu, {
        'origin': origin_3d,
        'spacing': spacing_3d,
        'shape': image_3d_hu.shape,
        'slice_z_positions': slice_z_positions,
        'slice_thickness': slice_thickness,
        # Full geometry, so masks can be mapped to patient coordinates.
        'image_position_patient': np.array(
            [[float(v) for v in s.ImagePositionPatient] for s in slices]),
        'image_orientation_patient': np.array(
            [float(v) for v in getattr(first, 'ImageOrientationPatient', [1, 0, 0, 0, 1, 0])]),
        'pixel_spacing_rc': (float(pixel_spacing[0]), float(pixel_spacing[1])),
        # Identifiers for referencing this CT from an RTSTRUCT.
        'sop_instance_uids': [str(s.SOPInstanceUID) for s in slices],
        'sop_class_uid': str(first.SOPClassUID),
        'series_instance_uid': str(getattr(first, 'SeriesInstanceUID', '')),
        'frame_of_reference_uid': str(getattr(first, 'FrameOfReferenceUID', '')),
        'patient_study': {
            kw: first.data_element(kw).value
            for kw in _PATIENT_STUDY_KEYWORDS if kw in first
        },
        'other_series_ignored': other_series_ignored,
        'duplicate_slices_dropped': duplicate_slices_dropped,
    }

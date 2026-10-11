import numpy as np
import os
import pydicom

def load_dicom_series_to_hu(dicom_dir):
    """
    Load a DICOM series from a directory and convert to Hounsfield Units.
    
    Args:
        dicom_dir: Path to directory containing DICOM files
        
    Returns:
        tuple: (3D numpy array of HU values, dict with spatial metadata)
    """
    slices = []
    for s in os.listdir(dicom_dir):
        try:
            filepath = os.path.join(dicom_dir, s)
            ds = pydicom.dcmread(filepath)
            if ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.2' and 'PixelData' in ds:
                slices.append(ds)
        except (pydicom.errors.InvalidDicomError, AttributeError, KeyError):
            # Skip non-DICOM files or files with missing required attributes
            continue
    
    if not slices:
        print(f"No valid CT DICOM images found in {dicom_dir}")
        return None, None
    
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    
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
    
    return image_3d_hu, {
        'origin': origin_3d,
        'spacing': spacing_3d,
        'shape': image_3d_hu.shape,
        'slice_z_positions': slice_z_positions,
        'slice_thickness': slice_thickness
    }



import streamlit as st
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import nibabel as nib

from dicom_utils import load_dicom_series_to_hu, create_metal_mask_from_rtstruct
from metal_detection import (detect_metal_volume, create_affine_from_dicom_meta, 
                            save_mask_as_nifti, detect_metal_adaptive)
from metal_detection_v3 import detect_metal_adaptive_3d
from contour_operations import (create_bright_artifact_mask, create_dark_artifact_mask,
                               create_bone_mask, save_all_contours_as_nifti, refine_mask,
                               create_russian_doll_segmentation)
from visualization import (create_overlay_image, create_histogram, fig_to_base64, 
                          create_multi_slice_view, visualize_star_profiles,
                          plot_threshold_evolution, visualize_discrimination_slice)

# Page configuration
st.set_page_config(
    page_title="CT Metal Artifact Characterization",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'ct_volume' not in st.session_state:
    st.session_state.ct_volume = None
    st.session_state.ct_metadata = None
    st.session_state.current_slice = 0
    st.session_state.masks = {}
    st.session_state.selected_patient = None
    st.session_state.metal_detection_result = None
    st.session_state.affine = None

# Header
st.title("🏥 CT Metal Artifact Characterization")
st.markdown("Advanced segmentation with automatic metal detection and boolean operations")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # Patient selection
    data_dir = Path("../data")
    if not data_dir.exists():
        data_dir = Path("data")  # Fallback for different run contexts
    
    patient_dirs = [d for d in data_dir.iterdir() if d.is_dir() and "HIP" in d.name] if data_dir.exists() else []
    patient_names = sorted([d.name for d in patient_dirs])
    
    selected_patient = st.selectbox(
        "Select Patient",
        patient_names,
        help="Choose a patient dataset to analyze"
    )
    
    if selected_patient != st.session_state.selected_patient:
        # Reset state when patient changes
        st.session_state.ct_volume = None
        st.session_state.ct_metadata = None
        st.session_state.masks = {}
        st.session_state.metal_detection_result = None
        st.session_state.selected_patient = selected_patient
    
    # Load data button
    if st.button("Load Patient Data", type="primary"):
        with st.spinner("Loading DICOM data..."):
            patient_path = data_dir / selected_patient
            
            # Find CT directory
            ct_dirs = [d for d in patient_path.iterdir() if d.is_dir() and "CT" in d.name]
            if ct_dirs:
                ct_dir = ct_dirs[0]
                ct_volume, ct_metadata = load_dicom_series_to_hu(str(ct_dir))
                
                if ct_volume is not None:
                    st.session_state.ct_volume = ct_volume
                    st.session_state.ct_metadata = ct_metadata
                    st.session_state.current_slice = ct_volume.shape[0] // 2
                    st.session_state.affine = create_affine_from_dicom_meta(ct_metadata)
                    st.success(f"Loaded {ct_volume.shape[0]} slices successfully!")
                else:
                    st.error("Failed to load CT data")
            else:
                st.error("No CT directory found for this patient")
    
    st.markdown("---")
    
    # Analysis parameters
    st.subheader("Analysis Parameters")
    
    analysis_tab1, analysis_tab2 = st.tabs(["Metal Detection", "Artifact Thresholds"])
    
    with analysis_tab1:
        st.markdown("**Advanced Metal Detection**")
        st.info("💡 3D adaptive method combines coronal/sagittal analysis with star profile algorithm")
        
        detection_method = st.radio(
            "Detection Method",
            ["3D Adaptive + Star Algorithm (Recommended)", "Legacy with Initial Threshold"],
            help="3D Adaptive: Combines 3D coronal/sagittal analysis with star profile algorithm. Legacy: Uses initial HU threshold."
        )
        
        if detection_method == "3D Adaptive + Star Algorithm (Recommended)":
            margin_cm = st.slider(
                "3D Search Margin (cm)",
                min_value=1.0,
                max_value=5.0,
                value=2.0,
                step=0.5,
                help="Margin around detected metal extent in all planes"
            )
            
            fw_percentage = st.slider(
                "Full Width Percentage",
                min_value=50,
                max_value=90,
                value=75,
                step=5,
                help="Percentage of peak for threshold detection (lower = more inclusive)"
            )
            
            intensity_percentile = st.slider(
                "Initial Detection Percentile",
                min_value=99.0,
                max_value=99.9,
                value=99.5,
                step=0.1,
                help="Top percentile of voxels to use for initial detection"
            )
            
        else:
            # Legacy parameters
            roi_margin_cm = st.slider(
                "ROI Margin (cm)",
                min_value=1.0,
                max_value=10.0,
                value=5.0,
                step=0.5,
                help="Margin around detected metal in centimeters"
            )
            
            min_metal_hu = st.number_input(
                "Initial Metal Threshold (HU)",
                value=2000,
                min_value=1000,
                max_value=4000,
                step=100,
                help="Minimum HU to detect metal cluster"
            )
            
            fw_percentage = st.slider(
                "Full Width Percentage",
                min_value=50,
                max_value=90,
                value=60,
                step=5,
                help="Percentage of peak for threshold detection"
            )
            
            dilation_iterations = st.slider(
                "Metal Region Connection",
                min_value=1,
                max_value=10,
                value=5,
                help="Dilation iterations to connect nearby metal regions"
            )
    
    with analysis_tab2:
        st.markdown("**Artifact Segmentation Method**")
        
        segmentation_method = st.radio(
            "Segmentation Approach",
            ["Russian Doll with Smart Discrimination (Recommended)", 
             "Russian Doll with Enhanced Edge Analysis", 
             "Legacy Threshold-Based"],
            help="Russian Doll: Fast distance-based discrimination. Enhanced: Edge coherence and structural analysis. Legacy: Simple threshold-based."
        )
        
        if segmentation_method == "Russian Doll with Smart Discrimination (Recommended)":
            st.info("🧠 Uses distance-based analysis for fast bone/artifact discrimination")
            
            col1, col2 = st.columns(2)
            with col1:
                dark_high = st.number_input("Dark Artifact Max HU", value=-150)
                bone_low = st.number_input("Bone/Bright Min HU", value=300)
            with col2:
                bone_high = st.number_input("Bone/Bright Max HU", value=1500)
                artifact_distance_cm = st.slider(
                    "Max Artifact Distance from Metal (cm)",
                    min_value=2.0,
                    max_value=15.0,
                    value=10.0,
                    step=1.0,
                    help="Maximum distance from metal to consider for bright artifacts"
                )
            
            # Not used in Russian doll method but keep for compatibility
            bright_low = bone_low
            bright_high = 3000
            
        elif segmentation_method == "Russian Doll with Enhanced Edge Analysis":
            st.info("🔬 Advanced edge coherence analysis for superior bone/artifact discrimination")
            st.warning("⚠️ This method is slower but provides better accuracy for challenging cases")
            
            col1, col2 = st.columns(2)
            with col1:
                dark_high = st.number_input("Dark Artifact Max HU", value=-150, key="enh_dark")
                bone_low = st.number_input("Bone/Bright Min HU", value=300, key="enh_bone_low")
            with col2:
                bone_high = st.number_input("Bone/Bright Max HU", value=1500, key="enh_bone_high")
                artifact_distance_cm = st.slider(
                    "Max Analysis Distance (cm)",
                    min_value=2.0,
                    max_value=15.0,
                    value=10.0,
                    step=1.0,
                    help="Maximum distance from metal to analyze",
                    key="enh_dist"
                )
            
            st.markdown("**Enhanced Features:**")
            st.markdown("- Edge coherence analysis (bone has continuous edges)")
            st.markdown("- Gradient jump detection (bone has sharp transitions)")
            st.markdown("- Radial vs tangential feature analysis")
            st.markdown("- 3D structural continuity tracking")
            st.markdown("- Multi-scale edge persistence")
            
            # Not used in Russian doll method but keep for compatibility
            bright_low = bone_low
            bright_high = 3000
            
        else:
            # Legacy parameters
            col1, col2 = st.columns(2)
            with col1:
                bright_low = st.number_input("Bright Artifact Min HU", value=800)
                dark_high = st.number_input("Dark Artifact Max HU", value=-200)
            with col2:
                bright_high = st.number_input("Bright Artifact Max HU", value=3000)
                bone_low = st.number_input("Bone Min HU", value=150)
            
            bone_high = st.number_input("Bone Max HU", value=1500)
            artifact_distance_cm = 10.0  # Default for compatibility
    
    st.markdown("---")
    
    # Contour Display Settings
    st.subheader("Contour Display")
    
    # Initialize contour visibility state
    if 'contour_visibility' not in st.session_state:
        st.session_state.contour_visibility = {
            'metal': True,
            'bright_artifacts': True,
            'dark_artifacts': True,
            'bone': True
        }
    
    # Initialize custom names
    if 'contour_names' not in st.session_state:
        st.session_state.contour_names = {
            'metal': 'Metal Implant',
            'bright_artifacts': 'Bright Artifacts',
            'dark_artifacts': 'Dark Artifacts',
            'bone': 'Bone'
        }
    
    # Contour visibility toggles
    st.markdown("**Visibility**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.contour_visibility['metal'] = st.checkbox(
            "Metal", value=st.session_state.contour_visibility['metal'], key="vis_metal")
        st.session_state.contour_visibility['bright_artifacts'] = st.checkbox(
            "Bright Artifacts", value=st.session_state.contour_visibility['bright_artifacts'], key="vis_bright")
    
    with col2:
        st.session_state.contour_visibility['dark_artifacts'] = st.checkbox(
            "Dark Artifacts", value=st.session_state.contour_visibility['dark_artifacts'], key="vis_dark")
        st.session_state.contour_visibility['bone'] = st.checkbox(
            "Bone", value=st.session_state.contour_visibility['bone'], key="vis_bone")
    
    # Contour name editing
    if st.checkbox("Edit Contour Names"):
        st.markdown("**Custom Names**")
        for key in ['metal', 'bright_artifacts', 'dark_artifacts', 'bone']:
            st.session_state.contour_names[key] = st.text_input(
                f"{key.replace('_', ' ').title()} Name:", 
                value=st.session_state.contour_names[key],
                key=f"name_{key}"
            )
    
    st.markdown("---")
    
    # Export options
    st.subheader("Export Options")
    output_format = st.selectbox(
        "Output Format",
        ["NIFTI (.nii.gz)", "Multi-label NIFTI", "Separate Binary Masks", "DICOM RT Structure"],
        help="Choose export format for contours"
    )
    
    if st.button("Export All Contours", disabled=not st.session_state.masks):
        if st.session_state.masks:
            with st.spinner("Exporting contours..."):
                patient_name = selected_patient.replace(" ", "_")
                output_dir = Path("../output") / patient_name
                if not output_dir.parent.exists():
                    output_dir = Path("output") / patient_name
                output_dir.mkdir(exist_ok=True, parents=True)
                
                if output_format == "DICOM RT Structure":
                    # DICOM export
                    from dicom_export import create_rtstruct_from_masks, save_rtstruct
                    
                    # Prepare patient info
                    patient_info = {
                        'PatientName': patient_name,
                        'PatientID': patient_name.split('_')[0] if '_' in patient_name else patient_name
                    }
                    
                    # Create RT Structure Set
                    rtstruct = create_rtstruct_from_masks(
                        st.session_state.masks,
                        st.session_state.ct_metadata,
                        patient_info,
                        st.session_state.contour_names
                    )
                    
                    # Save RT Structure
                    output_path = output_dir / f"{patient_name}_RTSTRUCT.dcm"
                    save_rtstruct(rtstruct, str(output_path))
                    st.success(f"Exported DICOM RT Structure to {output_path}")
                
                else:
                    # NIFTI export
                    if st.session_state.affine is not None:
                        output_prefix = str(output_dir / patient_name)
                        save_all_contours_as_nifti(
                            st.session_state.masks, 
                            st.session_state.affine,
                            output_prefix
                        )
                        st.success(f"Exported NIFTI contours to {output_dir}")
                    else:
                        st.error("NIFTI export requires affine transformation matrix")

# Main content area
if st.session_state.ct_volume is not None:
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Single Slice Analysis", "Multi-Slice View", 
                                       "Metal Detection Details", "Statistics"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("CT Slice Viewer")
            
            # Slice selector
            max_slice = st.session_state.ct_volume.shape[0] - 1
            current_slice = st.slider(
                "Select Slice",
                min_value=0,
                max_value=max_slice,
                value=st.session_state.current_slice,
                format="Slice %d"
            )
            st.session_state.current_slice = current_slice
            
            # Get current slice data
            ct_slice = st.session_state.ct_volume[current_slice]
            
            # Display slice info
            z_pos = st.session_state.ct_metadata['slice_z_positions'][current_slice]
            st.info(f"Slice {current_slice + 1} of {max_slice + 1} | Z position: {z_pos:.2f} mm")
            
            # Analysis buttons
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("🎯 Detect Metal Automatically", type="primary"):
                    with st.spinner("Detecting metal implant..."):
                        if detection_method == "3D Adaptive + Star Algorithm (Recommended)":
                            # 3D adaptive method with coronal/sagittal analysis + star profiles
                            result = detect_metal_adaptive_3d(
                                st.session_state.ct_volume,
                                st.session_state.ct_metadata['spacing'],
                                fw_percentage=fw_percentage,
                                margin_cm=margin_cm,
                                intensity_percentile=intensity_percentile
                            )
                            
                            # roi_bounds is already in the result as roi_bounds
                        else:
                            # Legacy method with initial threshold
                            result = detect_metal_volume(
                                st.session_state.ct_volume,
                                st.session_state.ct_metadata['spacing'],
                                margin_cm=roi_margin_cm,
                                fw_percentage=fw_percentage,
                                min_metal_hu=min_metal_hu,
                                dilation_iterations=dilation_iterations
                            )
                        
                        if result['mask'] is not None and np.any(result['mask']):
                            st.session_state.metal_detection_result = result
                            st.session_state.masks['metal'] = result['mask']
                            
                            metal_count = np.sum(result['mask'])
                            if detection_method == "3D Adaptive + Star Algorithm (Recommended)":
                                st.success(f"3D adaptive + star algorithm detection complete! Found {metal_count:,} metal voxels")
                                if 'analysis' in result and result['analysis']:
                                    thresh = result['analysis']['threshold_used']
                                    extent = result['analysis']['extent_voxels']
                                    st.info(f"Auto-detected threshold: {thresh:.0f} HU")
                                    st.info(f"3D extent: {extent['z']}×{extent['y']}×{extent['x']} voxels")
                                if 'individual_regions' in result:
                                    total_regions = sum(len(regions) for regions in result['individual_regions'].values())
                                    st.info(f"Created {total_regions} focused ROI regions across {len(result['individual_regions'])} slices")
                            else:
                                st.success(f"Legacy detection complete! Found {metal_count:,} metal voxels")
                        else:
                            st.error("No metal implant detected")
            
            with col_btn2:
                if st.button("🔍 Segment All Artifacts", 
                           disabled='metal' not in st.session_state.masks):
                    if 'metal' in st.session_state.masks:
                        with st.spinner("Segmenting artifacts..."):
                            metal_mask = st.session_state.masks['metal']
                            roi_bounds = st.session_state.metal_detection_result['roi_bounds']
                            
                            if segmentation_method == "Russian Doll with Smart Discrimination (Recommended)":
                                # Use the smart Russian doll segmentation
                                with st.spinner("Running smart bone/artifact discrimination..."):
                                    segmentation_result = create_russian_doll_segmentation(
                                        st.session_state.ct_volume,
                                        metal_mask,
                                        st.session_state.ct_metadata['spacing'],
                                        roi_bounds,
                                        dark_threshold_high=dark_high,
                                        bone_threshold_low=bone_low,
                                        bone_threshold_high=bone_high,
                                        bright_artifact_max_distance_cm=artifact_distance_cm,
                                        use_fast_mode=True,
                                        use_enhanced_mode=False
                                    )
                                
                                # Update masks - this was missing for fast mode!
                                if segmentation_result:
                                    for mask_name in ['dark_artifacts', 'bone', 'bright_artifacts']:
                                        mask = segmentation_result.get(mask_name)
                                        if mask is not None:
                                            st.session_state.masks[mask_name] = mask.astype(bool) if hasattr(mask, 'astype') else mask
                                    
                            elif segmentation_method == "Russian Doll with Enhanced Edge Analysis":
                                # Use enhanced edge-based discrimination
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                def update_progress(progress, message):
                                    progress_bar.progress(progress)
                                    status_text.text(message)
                                
                                try:
                                    segmentation_result = create_russian_doll_segmentation(
                                        st.session_state.ct_volume,
                                        metal_mask,
                                        st.session_state.ct_metadata['spacing'],
                                        roi_bounds,
                                        dark_threshold_high=dark_high,
                                        bone_threshold_low=bone_low,
                                        bone_threshold_high=bone_high,
                                        bright_artifact_max_distance_cm=artifact_distance_cm,
                                        use_fast_mode=False,
                                        use_enhanced_mode=True,
                                        progress_callback=update_progress
                                    )
                                except Exception as e:
                                    progress_bar.empty()
                                    status_text.empty()
                                    st.error(f"Enhanced segmentation failed: {str(e)}")
                                    st.exception(e)
                                    segmentation_result = None
                                
                                # Clear progress indicators
                                progress_bar.empty()
                                status_text.empty()
                                
                                if segmentation_result:
                                    # Update masks for both Russian doll methods - ensure they're boolean
                                    for mask_name in ['dark_artifacts', 'bone', 'bright_artifacts']:
                                        mask = segmentation_result.get(mask_name)
                                        if mask is not None:
                                            # Ensure mask is boolean
                                            st.session_state.masks[mask_name] = mask.astype(bool) if hasattr(mask, 'astype') else mask
                                    
                                    
                                else:
                                    st.warning("Segmentation returned no results")
                            
                            # Common code for both Russian doll methods (moved outside enhanced-only block)
                            if segmentation_result and segmentation_method.startswith("Russian Doll"):
                                # Store additional results for analysis
                                if 'segmentation_info' not in st.session_state:
                                    st.session_state.segmentation_info = {}
                                st.session_state.segmentation_info['confidence_map'] = segmentation_result.get('confidence_map')
                                st.session_state.segmentation_info['distance_map'] = segmentation_result.get('distance_map')
                                
                                if segmentation_method == "Russian Doll with Enhanced Edge Analysis":
                                    st.success("Enhanced edge-based segmentation complete!")
                                else:
                                    st.success("Smart artifact segmentation complete!")
                                
                                # Show statistics
                                bone_voxels = np.sum(st.session_state.masks['bone']) if 'bone' in st.session_state.masks else 0
                                bright_voxels = np.sum(st.session_state.masks['bright_artifacts']) if 'bright_artifacts' in st.session_state.masks else 0
                                st.info(f"Discriminated {bone_voxels:,} bone voxels from {bright_voxels:,} bright artifact voxels")
                                
                            else:
                                # Legacy method
                                bright_mask = create_bright_artifact_mask(
                                    st.session_state.ct_volume,
                                    metal_mask,
                                    roi_bounds,
                                    bright_low,
                                    bright_high
                                )
                                
                                dark_mask = create_dark_artifact_mask(
                                    st.session_state.ct_volume,
                                    metal_mask,
                                    roi_bounds,
                                    dark_high
                                )
                                
                                bone_mask = create_bone_mask(
                                    st.session_state.ct_volume,
                                    metal_mask,
                                    bright_mask,
                                    dark_mask,
                                    roi_bounds,
                                    bone_low,
                                    bone_high
                                )
                                
                                # Refine masks
                                st.session_state.masks['bright_artifacts'] = refine_mask(bright_mask)
                                st.session_state.masks['dark_artifacts'] = refine_mask(dark_mask)
                                st.session_state.masks['bone'] = refine_mask(bone_mask)
                                
                                st.success("Legacy artifact segmentation complete!")
            
            # Display visualization
            if st.session_state.masks:
                roi_bounds = None
                if st.session_state.metal_detection_result:
                    roi_bounds = st.session_state.metal_detection_result['roi_bounds']
                    # Convert 3D bounds to 2D for current slice
                    roi_bounds_2d = {
                        'y_min': roi_bounds['y_min'],
                        'y_max': roi_bounds['y_max'], 
                        'x_min': roi_bounds['x_min'],
                        'x_max': roi_bounds['x_max']
                    }
                
                # Create masks dict for current slice only - respect visibility settings
                slice_masks = {}
                for name, mask in st.session_state.masks.items():
                    if isinstance(mask, np.ndarray) and mask.ndim == 3:
                        # Only include if visibility is enabled
                        if st.session_state.contour_visibility.get(name, True):
                            slice_masks[name] = mask[current_slice]
                
                # Convert roi_bounds_2d dict to tuple format expected by create_overlay_image
                roi_boundaries_tuple = None
                if roi_bounds_2d:
                    roi_boundaries_tuple = (
                        roi_bounds_2d['y_min'],
                        roi_bounds_2d['y_max'],
                        roi_bounds_2d['x_min'],
                        roi_bounds_2d['x_max']
                    )
                
                # Get individual regions for this slice if using 3D adaptive detection
                current_slice_regions = None
                if (st.session_state.metal_detection_result and 
                    'individual_regions' in st.session_state.metal_detection_result and
                    current_slice in st.session_state.metal_detection_result['individual_regions']):
                    current_slice_regions = st.session_state.metal_detection_result['individual_regions'][current_slice]
                
                fig = create_overlay_image(
                    ct_slice,
                    slice_masks,
                    roi_boundaries_tuple,
                    current_slice,
                    individual_regions=current_slice_regions,
                    custom_names=st.session_state.contour_names
                )
                st.pyplot(fig)
                plt.close()
            else:
                # Show simple preview
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(ct_slice, cmap='gray', vmin=-150, vmax=250)
                ax.set_title(f"CT Slice {current_slice}")
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
        
        with col2:
            st.subheader("Analysis Results")
            
            if st.session_state.metal_detection_result:
                # Show detected thresholds
                st.markdown("**Adaptive Thresholds**")
                thresholds = st.session_state.metal_detection_result['slice_thresholds']
                current_thresh = next((t for t in thresholds if t['slice'] == current_slice), None)
                
                if current_thresh and current_thresh['thresholds']:
                    lower, upper = current_thresh['thresholds']
                    st.text(f"Metal: {lower:.0f} - {upper:.0f} HU")
                else:
                    st.text("Metal: Default thresholds")
                
                st.text(f"Bright: {bright_low} - {bright_high} HU")
                st.text(f"Dark: < {dark_high} HU")
                st.text(f"Bone: {bone_low} - {bone_high} HU")
            
            # Display pixel counts
            if st.session_state.masks:
                st.markdown("**Segmentation Statistics**")
                for mask_name, mask in st.session_state.masks.items():
                    if isinstance(mask, np.ndarray):
                        if mask.ndim == 3:
                            count = np.sum(mask[current_slice])
                            total = np.sum(mask)
                            st.text(f"{mask_name}: {count:,} pixels (slice) / {total:,} voxels (total)")
                        else:
                            count = np.sum(mask)
                            st.text(f"{mask_name}: {count:,} pixels")
            
            # Show histograms
            if st.session_state.masks and st.checkbox("Show Intensity Histograms"):
                st.markdown("**Intensity Distributions**")
                
                colors = {
                    'metal': 'red',
                    'bright_artifacts': 'yellow',
                    'dark_artifacts': 'magenta',
                    'bone': 'blue'
                }
                
                for mask_name, mask in st.session_state.masks.items():
                    if mask_name in colors and isinstance(mask, np.ndarray):
                        if mask.ndim == 3:
                            mask_slice = mask[current_slice]
                        else:
                            mask_slice = mask
                        
                        hu_values = ct_slice[mask_slice]
                        if hu_values.size > 0:
                            fig = create_histogram(
                                hu_values,
                                mask_name.replace('_', ' ').title(),
                                colors[mask_name]
                            )
                            if fig:
                                st.pyplot(fig)
                                plt.close()
    
    with tab2:
        st.subheader("Multi-Slice Overview")
        
        if st.session_state.masks:
            # Select slices to display
            n_slices = st.slider("Number of slices to display", 4, 16, 8)
            
            # Get slice indices
            if st.session_state.metal_detection_result:
                roi_bounds = st.session_state.metal_detection_result['roi_bounds']
                z_min, z_max = roi_bounds['z_min'], roi_bounds['z_max']
                slice_indices = np.linspace(z_min, z_max-1, n_slices, dtype=int)
            else:
                slice_indices = np.linspace(0, st.session_state.ct_volume.shape[0]-1, 
                                          n_slices, dtype=int)
            
            # Create multi-slice view
            individual_regions = None
            if (st.session_state.metal_detection_result and 
                'individual_regions' in st.session_state.metal_detection_result):
                individual_regions = st.session_state.metal_detection_result['individual_regions']
            
            # Filter masks based on visibility settings
            visible_masks = {}
            for name, mask in st.session_state.masks.items():
                if st.session_state.contour_visibility.get(name, True):
                    visible_masks[name] = mask
            
            fig = create_multi_slice_view(
                st.session_state.ct_volume,
                visible_masks,
                slice_indices,
                roi_bounds if st.session_state.metal_detection_result else None,
                individual_regions=individual_regions
            )
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Run metal detection and segmentation first to view multiple slices")
    
    with tab3:
        st.subheader("Metal Detection Analysis")
        
        if st.session_state.metal_detection_result:
            result = st.session_state.metal_detection_result
            
            # Show star profile visualization
            if st.checkbox("Show Star Profile Analysis"):
                current_slice = st.session_state.current_slice
                
                # Check if we have individual regions for this slice
                if ('individual_regions' in result and 
                    current_slice in result['individual_regions'] and
                    result['individual_regions'][current_slice]):
                    
                    # Use the first component's bounds for this slice
                    component = result['individual_regions'][current_slice][0]
                    roi_bounds_2d = {
                        'y_min': component['y_min'],
                        'y_max': component['y_max'],
                        'x_min': component['x_min'],
                        'x_max': component['x_max']
                    }
                    center_y = component['center_y']
                    center_x = component['center_x']
                    
                    # Get the slice result with profiles if available
                    slice_results = result.get('slice_thresholds', [])
                    current_slice_result = next((r for r in slice_results if r['slice'] == current_slice), None)
                    
                    if current_slice_result:
                        # For now, generate synthetic profiles for visualization
                        # Since the 3D method doesn't store the actual profiles
                        from metal_detection_v3 import get_star_profile_lines
                        
                        profiles = get_star_profile_lines(
                            st.session_state.ct_volume[current_slice],
                            center_y,
                            center_x,
                            roi_bounds_2d
                        )
                        
                        fig = visualize_star_profiles(
                            st.session_state.ct_volume[current_slice],
                            profiles,
                            (center_y, center_x),
                            roi_bounds_2d,
                            current_slice_result['thresholds']
                        )
                        st.pyplot(fig)
                        plt.close()
                    else:
                        st.info("No profile data available for this slice")
                else:
                    st.info("Star profile visualization requires individual region detection")
            
            # Show threshold evolution
            if st.checkbox("Show Threshold Evolution Across Slices"):
                fig = plot_threshold_evolution(result['slice_thresholds'])
                st.pyplot(fig)
                plt.close()
            
            # Detection summary
            st.markdown("**Detection Summary**")
            center = result['center_coords']
            st.text(f"Metal center: ({center[0]}, {center[1]}, {center[2]})")
            st.text(f"ROI: Z [{roi_bounds['z_min']}-{roi_bounds['z_max']}], "
                   f"Y [{roi_bounds['y_min']}-{roi_bounds['y_max']}], "
                   f"X [{roi_bounds['x_min']}-{roi_bounds['x_max']}]")
        else:
            st.info("Run automatic metal detection to see detailed analysis")
    
    with tab4:
        st.subheader("Volume Statistics")
        
        if st.session_state.masks:
            # Calculate volumes
            spacing = st.session_state.ct_metadata['spacing']
            voxel_volume = np.prod(spacing) / 1000  # Convert to cm³
            
            st.markdown("**Tissue Volumes**")
            
            data = []
            for mask_name, mask in st.session_state.masks.items():
                if isinstance(mask, np.ndarray):
                    voxel_count = np.sum(mask)
                    volume_cm3 = voxel_count * voxel_volume
                    
                    data.append({
                        'Tissue Type': mask_name.replace('_', ' ').title(),
                        'Voxel Count': f"{voxel_count:,}",
                        'Volume (cm³)': f"{volume_cm3:.2f}",
                        'Percentage': f"{100 * voxel_count / mask.size:.2f}%"
                    })
            
            if data:
                import pandas as pd
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
            
            # Show HU statistics
            if st.checkbox("Show HU Statistics by Region"):
                st.markdown("**Hounsfield Unit Statistics**")
                
                for mask_name, mask in st.session_state.masks.items():
                    if isinstance(mask, np.ndarray):
                        hu_values = st.session_state.ct_volume[mask]
                        if hu_values.size > 0:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric(f"{mask_name} Mean", f"{np.mean(hu_values):.0f} HU")
                            with col2:
                                st.metric("Std Dev", f"{np.std(hu_values):.0f}")
                            with col3:
                                st.metric("Min", f"{np.min(hu_values):.0f} HU")
                            with col4:
                                st.metric("Max", f"{np.max(hu_values):.0f} HU")
            
            # Show discrimination confidence if available
            if ('segmentation_info' in st.session_state and 
                'confidence_map' in st.session_state.segmentation_info and
                st.checkbox("Show Discrimination Confidence")):
                st.markdown("**Bone vs Artifact Discrimination Confidence**")
                
                confidence_map = st.session_state.segmentation_info['confidence_map']
                confident_voxels = confidence_map > 0
                
                if np.any(confident_voxels):
                    avg_confidence = np.mean(confidence_map[confident_voxels])
                    high_confidence = np.sum(confidence_map > 0.8)
                    medium_confidence = np.sum((confidence_map > 0.5) & (confidence_map <= 0.8))
                    low_confidence = np.sum((confidence_map > 0) & (confidence_map <= 0.5))
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
                    with col2:
                        st.metric("High Confidence", f"{high_confidence:,} voxels")
                    with col3:
                        st.metric("Low Confidence", f"{low_confidence:,} voxels")
                    
                    # Show confidence distribution  
                    fig, ax = plt.subplots(figsize=(8, 4))
                    confidence_values = confidence_map[confident_voxels]
                    ax.hist(confidence_values, bins=50, color='purple', alpha=0.7, edgecolor='black')
                    ax.set_xlabel('Confidence Score')
                    ax.set_ylabel('Number of Voxels')
                    ax.set_title('Discrimination Confidence Distribution')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
                    
                # Show discrimination visualization for current slice
                if st.checkbox("Show Discrimination Visualization for Current Slice"):
                    current_slice = st.session_state.current_slice
                    
                    # Get the masks for the current slice
                    bone_slice = st.session_state.masks.get('bone', np.zeros_like(ct_volume[current_slice], dtype=bool))
                    artifact_slice = st.session_state.masks.get('bright_artifacts', np.zeros_like(ct_volume[current_slice], dtype=bool))
                    
                    if bone_slice.ndim == 3:
                        bone_slice = bone_slice[current_slice]
                    if artifact_slice.ndim == 3:
                        artifact_slice = artifact_slice[current_slice]
                    
                    # Get confidence for this slice
                    conf_slice = confidence_map[current_slice] if confidence_map.ndim == 3 else confidence_map
                    
                    # Create visualization
                    fig_disc = visualize_discrimination_slice(
                        st.session_state.ct_volume[current_slice],
                        bone_slice,
                        artifact_slice,
                        conf_slice,
                        current_slice
                    )
                    st.pyplot(fig_disc)
                    plt.close()
        else:
            st.info("Perform segmentation to see volume statistics")

else:
    # No data loaded
    st.info("👈 Please select a patient and load data from the sidebar to begin analysis")
    
    # Show instructions
    with st.expander("Quick Start Guide", expanded=True):
        st.markdown("""
        ### How to use this application:
        
        1. **Load Data**: Select a patient from the sidebar and click "Load Patient Data"
        2. **Detect Metal**: Click "Detect Metal Automatically" to find the implant using FW75% thresholding
        3. **Segment Artifacts**: Click "Segment All Artifacts" to identify bright/dark artifacts and bone
        4. **Explore Results**: 
           - Use the tabs to view different analysis modes
           - Adjust thresholds in the sidebar as needed
           - Export results as NIFTI files
        
        ### Key Features:
        - **Automatic Metal Detection**: Uses 16-point star profiles and FW75% maximum thresholding
        - **Boolean Operations**: Subtracts metal from artifact regions for accurate segmentation
        - **Adaptive Thresholding**: Each slice uses its own optimized thresholds
        - **NIFTI Export**: Save contours for machine learning pipelines
        """)
    
    # Display available patients
    st.subheader("Available Patients")
    data_dir = Path("../data")
    if not data_dir.exists():
        data_dir = Path("data")
    
    patient_dirs = [d for d in data_dir.iterdir() if d.is_dir() and "HIP" in d.name] if data_dir.exists() else []
    
    if patient_dirs:
        cols = st.columns(3)
        for i, patient_dir in enumerate(sorted(patient_dirs)):
            with cols[i % 3]:
                dcm_files = list(patient_dir.rglob('*.dcm'))
                st.metric(patient_dir.name, f"{len(dcm_files)} DICOM files")
    else:
        st.warning("No patient data found in the data directory")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p style='color: #888'>CT Metal Artifact Characterization Tool | Advanced Medical Imaging Analysis</p>
    </div>
    """,
    unsafe_allow_html=True
)
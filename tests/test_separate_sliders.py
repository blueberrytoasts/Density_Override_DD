#!/usr/bin/env python3
"""
Test script to verify separate bright artifact and bone sliders functionality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock streamlit to avoid import error
class MockStreamlit:
    class session_state:
        thresholds = {}
        
sys.modules['streamlit'] = MockStreamlit()

from app.config import ThresholdConfig

def test_config():
    """Test that separate configurations exist"""
    print("Testing ThresholdConfig for separate bright artifact and bone ranges...")
    
    # Check that Russian Doll configs exist
    assert hasattr(ThresholdConfig, 'RUSSIAN_DOLL_BRIGHT_ARTIFACTS'), "Missing RUSSIAN_DOLL_BRIGHT_ARTIFACTS"
    assert hasattr(ThresholdConfig, 'RUSSIAN_DOLL_BONE'), "Missing RUSSIAN_DOLL_BONE"
    
    # Check bright artifacts config
    bright_config = ThresholdConfig.RUSSIAN_DOLL_BRIGHT_ARTIFACTS
    print(f"✓ Bright Artifacts Range: {bright_config.default_min} - {bright_config.default_max} HU")
    print(f"  Label: {bright_config.label}")
    print(f"  Help: {bright_config.help_text}")
    
    # Check bone config  
    bone_config = ThresholdConfig.RUSSIAN_DOLL_BONE
    print(f"✓ Bone Tissue Range: {bone_config.default_min} - {bone_config.default_max} HU")
    print(f"  Label: {bone_config.label}")
    print(f"  Help: {bone_config.help_text}")
    
    # Verify they are different
    assert bright_config.default_min != bone_config.default_min or bright_config.default_max != bone_config.default_max, \
        "Bright and bone ranges should be different"
    
    print("\n✅ All configuration tests passed!")
    return True

def test_function_signatures():
    """Test that functions accept separate parameters"""
    print("\nTesting function signatures...")
    
    from app.contour_operations import create_russian_doll_segmentation
    import inspect
    
    # Check create_russian_doll_segmentation signature
    sig = inspect.signature(create_russian_doll_segmentation)
    params = list(sig.parameters.keys())
    
    assert 'bone_threshold_low' in params, "Missing bone_threshold_low parameter"
    assert 'bone_threshold_high' in params, "Missing bone_threshold_high parameter"
    assert 'bright_threshold_low' in params, "Missing bright_threshold_low parameter"
    assert 'bright_threshold_high' in params, "Missing bright_threshold_high parameter"
    
    print("✓ create_russian_doll_segmentation has separate bright and bone parameters")
    
    # Check discrimination functions
    from app.artifact_discrimination_fast import create_fast_russian_doll_segmentation
    sig = inspect.signature(create_fast_russian_doll_segmentation)
    params = list(sig.parameters.keys())
    
    assert 'bone_range' in params, "Missing bone_range parameter"
    assert 'bright_range' in params, "Missing bright_range parameter"
    
    print("✓ create_fast_russian_doll_segmentation has separate bright and bone parameters")
    
    print("\n✅ All function signature tests passed!")
    return True

def test_default_values():
    """Test that default values are correctly set"""
    print("\nTesting default values...")
    
    # Test Russian Doll defaults
    defaults = ThresholdConfig.get_russian_doll_defaults()
    
    # These should now be separate
    print(f"Default bright_min: {defaults['bright_min']} HU")
    print(f"Default bright_max: {defaults['bright_max']} HU")
    print(f"Default bone_min: {defaults['bone_min']} HU")
    print(f"Default bone_max: {defaults['bone_max']} HU")
    
    # Verify bright artifacts go higher than bone
    assert defaults['bright_max'] >= defaults['bone_max'], \
        "Bright artifacts should extend to higher HU than bone"
    
    print("\n✅ Default value tests passed!")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Separate Bright Artifact and Bone Slider Implementation")
    print("=" * 60)
    
    try:
        test_config()
        test_function_signatures()
        test_default_values()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! Separate sliders are correctly implemented.")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
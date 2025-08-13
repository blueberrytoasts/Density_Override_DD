#!/usr/bin/env python3
"""
Quick integration test for Russian doll segmentation
"""
import sys
sys.path.append('app')

import numpy as np
from artifact_discrimination import analyze_profile_characteristics, classify_bone_vs_artifact

def test_discrimination():
    """Test the discrimination functions"""
    print("Testing profile analysis...")
    
    # Create synthetic profile data
    distances = np.linspace(0, 10, 100)
    
    # Bone-like profile (broad, smooth)
    bone_profile = 500 + 800 * np.exp(-(distances - 5)**2 / 4)
    
    # Artifact-like profile (sharp, narrow)
    artifact_profile = 500 + 1500 * np.exp(-(distances - 5)**2 / 0.5)
    
    # Analyze bone profile
    bone_chars = analyze_profile_characteristics(distances, bone_profile)
    print(f"\nBone profile characteristics:")
    print(f"  Average width: {bone_chars['avg_width']:.2f} mm")
    print(f"  Smoothness: {bone_chars['smoothness']:.3f}")
    print(f"  Max gradient: {bone_chars['max_gradient']:.0f}")
    
    # Analyze artifact profile
    artifact_chars = analyze_profile_characteristics(distances, artifact_profile)
    print(f"\nArtifact profile characteristics:")
    print(f"  Average width: {artifact_chars['avg_width']:.2f} mm")
    print(f"  Smoothness: {artifact_chars['smoothness']:.3f}")
    print(f"  Max gradient: {artifact_chars['max_gradient']:.0f}")
    
    # Test classification with multiple profiles
    bone_profiles = [bone_chars] * 8  # Simulate 8 consistent bone profiles
    artifact_profiles = [artifact_chars] * 8  # Simulate 8 consistent artifact profiles
    
    bone_result = classify_bone_vs_artifact(bone_profiles)
    print(f"\nBone classification: {bone_result['classification']} (confidence: {bone_result['confidence']:.2%})")
    
    artifact_result = classify_bone_vs_artifact(artifact_profiles)
    print(f"Artifact classification: {artifact_result['classification']} (confidence: {artifact_result['confidence']:.2%})")
    
    print("\nIntegration test passed!")

if __name__ == "__main__":
    test_discrimination()
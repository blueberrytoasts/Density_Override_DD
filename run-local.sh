#!/bin/bash

# Script to launch the CT Artifact Characterization Streamlit app locally

echo "Starting CT Artifact Characterization App (Local Mode)..."

# Add local bin to PATH for user installations
export PATH=$PATH:/config/home/dev/.local/bin

# Change to app directory
cd app

# Launch Streamlit locally
streamlit run main.py
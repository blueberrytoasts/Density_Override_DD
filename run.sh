#!/bin/bash

# Script to launch the CT Artifact Characterization Streamlit app

echo "Starting CT Artifact Characterization App..."

# Add local bin to PATH for user installations
export PATH=$PATH:/config/home/dev/.local/bin

# Change to app directory
cd app

# Launch Streamlit on specified address and port
streamlit run main.py \
    --server.address 192.168.1.11 \
    --server.port 4224 \
    --server.headless true \
    --browser.gatherUsageStats false
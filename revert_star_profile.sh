#!/bin/bash
# Revert star profile implementation to original percentile-based method

echo "Reverting to original metal detection (percentile-based)..."

if [ -f "app/core/metal_detection.py.before_star_profile" ]; then
    cp app/core/metal_detection.py.before_star_profile app/core/metal_detection.py
    echo "[SUCCESS] Reverted metal_detection.py"
else
    echo "[ERROR] Backup file not found: app/core/metal_detection.py.before_star_profile"
    exit 1
fi

# Revert main.py UI changes
cd app
git checkout main.py 2>/dev/null || echo "[WARNING] Could not revert main.py via git, manual revert may be needed"

echo ""
echo "Revert complete! The system is back to percentile-based metal detection."
echo "Restart the Streamlit app to see changes."

"""Make the existing ``app/`` modules importable from the PySide app.

The Streamlit code in ``app/`` uses *flat* imports (e.g. ``from dicom_utils
import ...``, ``from core.metal_detection import ...``) because Streamlit runs
with ``app/`` as the working directory. To reuse that exact same algorithm code
from here without copying it, we put ``app/`` on ``sys.path`` once.

Import this module before importing any ``app/`` module:

    from pyside_app import bootstrap  # noqa: F401  (side effect: sys.path)
    from dicom_utils import load_dicom_series_to_hu
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_APP_DIR = os.path.join(_REPO_ROOT, "app")

if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

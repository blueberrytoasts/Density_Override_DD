"""Make the ``algorithms/`` modules importable from the PySide app.

The code in ``algorithms/`` uses *flat* imports (e.g. ``from dicom_utils
import ...``, ``from core.metal_detection import ...``), a holdover from the
Streamlit app, which ran with that folder (then ``app/``) as the working directory. To reuse that exact same algorithm code
from here without copying it, we put ``algorithms/`` on ``sys.path`` once.

Import this module before importing any ``algorithms/`` module:

    from pyside_app import bootstrap  # noqa: F401  (side effect: sys.path)
    from dicom_utils import load_dicom_series_to_hu
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ALGORITHMS_DIR = os.path.join(_REPO_ROOT, "algorithms")

if _ALGORITHMS_DIR not in sys.path:
    sys.path.insert(0, _ALGORITHMS_DIR)

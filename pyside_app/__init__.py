"""PySide6 desktop application for CT metal artifact characterization.

This package is a desktop (Qt) front-end that reuses the existing, UI-agnostic
algorithm modules under ``app/core`` (metal detection, discrimination) and
``app/`` (DICOM I/O). The long-term goal is to replace the Streamlit UI with a
responsive native application; see ``docs/`` for the migration plan.
"""

__version__ = "0.1.0"

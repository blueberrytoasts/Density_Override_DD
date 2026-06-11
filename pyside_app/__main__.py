"""Entry point: ``python -m pyside_app``."""
import sys

from PySide6.QtWidgets import QApplication

from pyside_app.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

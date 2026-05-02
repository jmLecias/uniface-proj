"""UniFace Live - Real-time face detection and recognition application."""

from PyQt6.QtWidgets import QApplication

from main_window import MainWindow


def main():
    """Start the UniFace Live application."""
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()

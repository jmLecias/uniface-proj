"""Video panel widget for displaying video streams."""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


class VideoPanel(QWidget):
    """A widget that displays a video stream with title and status."""

    def __init__(self, name):
        super().__init__()
        self.name = name

        self.title_label = QLabel(f"<b>{name}</b>")
        self.title_label.setStyleSheet("color: white;")

        self.video_label = QLabel("No video")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background: black; color: white;")
        self.video_label.setMinimumSize(320, 240)

        self.status_label = QLabel("Stopped")
        self.status_label.setStyleSheet("color: #cccccc;")

        layout = QVBoxLayout()
        layout.addWidget(self.title_label)
        layout.addWidget(self.video_label, stretch=1)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

    def update_frame(self, image):
        """Update the displayed frame."""
        pixmap = QPixmap.fromImage(image).scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(pixmap)

    def set_status(self, status):
        """Update the status label."""
        self.status_label.setText(status)

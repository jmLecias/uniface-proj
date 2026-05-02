"""Main window for UniFace Live application."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from video_panel import VideoPanel
from video_worker import VideoWorker
from face_db import reload_database


class MainWindow(QMainWindow):
    """Main application window for UniFace Live."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("UniFace Live")
        self.setMinimumSize(1200, 780)

        self.sources = {}

        self.source_name_input = QLineEdit()
        self.source_name_input.setPlaceholderText("Source name")
        self.source_url_input = QLineEdit("http://192.168.254.100:8080/video")
        self.source_url_input.setPlaceholderText("rtsp://192.168.x.x/stream or http://...")

        self.add_source_button = QPushButton("Add Source")
        self.add_source_button.clicked.connect(self.add_source)

        self.start_all_button = QPushButton("Start All")
        self.start_all_button.clicked.connect(self.start_all_sources)

        self.stop_all_button = QPushButton("Stop All")
        self.stop_all_button.clicked.connect(self.stop_all_sources)
        self.stop_all_button.setEnabled(False)

        self.remove_source_button = QPushButton("Remove Source")
        self.remove_source_button.clicked.connect(self.remove_selected_source)
        self.remove_source_button.setEnabled(False)

        self.refresh_button = QPushButton("Refresh DB")
        self.refresh_button.clicked.connect(self.reload_database)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #ffffff;")

        self.source_list = QListWidget()
        self.source_list.currentItemChanged.connect(self.on_source_selected)
        self.source_list.setStyleSheet("color: white; background: #1b1f2a;")

        source_controls_layout = QHBoxLayout()
        source_controls_layout.addWidget(self.source_name_input)
        source_controls_layout.addWidget(self.source_url_input)
        source_controls_layout.addWidget(self.add_source_button)

        source_action_layout = QHBoxLayout()
        source_action_layout.addWidget(self.start_all_button)
        source_action_layout.addWidget(self.stop_all_button)
        source_action_layout.addWidget(self.remove_source_button)
        source_action_layout.addWidget(self.refresh_button)

        left_panel = QVBoxLayout()
        left_panel.addWidget(QLabel("Sources"))
        left_panel.addWidget(self.source_list, stretch=1)
        left_panel.addLayout(source_controls_layout)
        left_panel.addLayout(source_action_layout)
        left_panel.addSpacing(12)
        left_panel.addWidget(self.status_label)

        self.video_grid = QGridLayout()
        self.video_grid.setSpacing(12)

        self.video_container = QWidget()
        self.video_container.setLayout(self.video_grid)

        main_layout = QHBoxLayout()
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setFixedWidth(360)
        left_widget.setStyleSheet("background: #121523; color: white;")

        main_layout.addWidget(left_widget)
        main_layout.addWidget(self.video_container, stretch=1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def add_source(self):
        """Add a new video source."""
        name = self.source_name_input.text().strip()
        url = self.source_url_input.text().strip()
        if not name or not url:
            QMessageBox.warning(self, "Missing source", "Enter both a source name and URL.")
            return

        if name in self.sources:
            QMessageBox.warning(self, "Duplicate source", "A source with that name already exists.")
            return

        panel = VideoPanel(name)
        self.sources[name] = {
            "url": url,
            "worker": None,
            "panel": panel,
        }

        self.source_list.addItem(QListWidgetItem(name))
        self.update_video_layout()
        self.start_source(name)
        self.source_name_input.clear()

    def start_source(self, name):
        """Start video processing for a source."""
        source = self.sources.get(name)
        if not source:
            return

        worker = source.get("worker")
        if worker and worker.isRunning():
            return

        worker = VideoWorker(name, source["url"])
        worker.frame_ready.connect(self.on_frame_ready)
        worker.status.connect(self.on_status)
        worker.start()
        source["worker"] = worker
        source["panel"].set_status("Starting...")
        self.stop_all_button.setEnabled(True)

    def stop_source(self, name):
        """Stop video processing for a source."""
        source = self.sources.get(name)
        if not source:
            return

        worker = source.get("worker")
        if worker:
            worker.stop()
            source["worker"] = None
            source["panel"].set_status("Stopped")

    def start_all_sources(self):
        """Start all video sources."""
        for name in self.sources:
            self.start_source(name)

    def stop_all_sources(self):
        """Stop all video sources."""
        for name in self.sources:
            self.stop_source(name)
        self.stop_all_button.setEnabled(False)

    def update_video_layout(self):
        """Update the grid layout of video panels."""
        for i in reversed(range(self.video_grid.count())):
            widget = self.video_grid.itemAt(i).widget()
            if widget is not None:
                self.video_grid.removeWidget(widget)

        columns = 2
        for index, source_name in enumerate(self.sources):
            row = index // columns
            column = index % columns
            self.video_grid.addWidget(self.sources[source_name]["panel"], row, column)

    def reload_database(self):
        """Reload the face database."""
        self.status_label.setText("Reloading face database...")
        reload_database()
        self.status_label.setText("Face database refreshed")

    def remove_selected_source(self):
        """Remove the selected video source."""
        current_item = self.source_list.currentItem()
        if current_item is None:
            return

        source_name = current_item.text()
        source = self.sources.pop(source_name, None)
        if source is not None:
            worker = source.get("worker")
            if worker:
                worker.stop()
            panel = source.get("panel")
            if panel is not None:
                panel.setParent(None)

        self.source_list.takeItem(self.source_list.currentRow())
        self.update_video_layout()
        self.remove_source_button.setEnabled(self.source_list.currentItem() is not None)
        self.status_label.setText(f"Removed source: {source_name}")

    def on_frame_ready(self, source_name, image, face_data):
        """Handle a new frame from a video source."""
        source = self.sources.get(source_name)
        if not source:
            return

        source["panel"].update_frame(image)
        source["panel"].set_status(f"Faces: {len(face_data)}")
        self.status_label.setText(f"{source_name}: {len(face_data)} faces")

    def on_status(self, source_name, message):
        """Handle a status update from a video source."""
        source = self.sources.get(source_name)
        if source:
            source["panel"].set_status(message)
        self.status_label.setText(f"{source_name}: {message}")

    def on_source_selected(self, current, previous):
        """Handle selection of a source in the list."""
        self.remove_source_button.setEnabled(current is not None)
        if current:
            self.status_label.setText(f"Selected source: {current.text()}")

    def closeEvent(self, event):
        """Handle window close event."""
        self.stop_all_sources()
        event.accept()

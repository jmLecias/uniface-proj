import os
import socket
import time
import hashlib
import threading

import cv2
import numpy as np
from uniface import FaceAnalyzer
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
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

MAX_AGE = 2.0  # seconds before forgetting a track
DB_PATH = "face-db"
CACHE_FILE = os.path.join(DB_PATH, "face_cache.npy")
SIGNATURE_FILE = os.path.join(DB_PATH, "face_cache_sig.txt")

try:
    analyzer = FaceAnalyzer(providers=["CUDAExecutionProvider"])
except TypeError:
    analyzer = FaceAnalyzer()
except Exception:
    analyzer = FaceAnalyzer()


def get_db_signature(path=DB_PATH):
    file_data = []

    for root, _, files in os.walk(path):
        for f in files:
            if f in ["face_cache.npy", "face_cache_sig.txt"]:
                continue

            full_path = os.path.join(root, f)
            if os.path.isfile(full_path):
                mtime = os.path.getmtime(full_path)
                file_data.append(f"{full_path}:{mtime}")

    file_data.sort()
    return hashlib.md5("".join(file_data).encode()).hexdigest()


def load_face_database(path=DB_PATH):
    db = {}

    for person_name in os.listdir(path):
        person_path = os.path.join(path, person_name)

        if not os.path.isdir(person_path):
            continue

        db[person_name] = []

        for file in os.listdir(person_path):
            img_path = os.path.join(person_path, file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (480, 480))
            results = analyzer.analyze(img)

            if len(results) == 0:
                print(f"⚠️ No face found in {img_path}")
                continue

            face = results[0]
            if hasattr(face, "embedding"):
                db[person_name].append(face.embedding)
                print(f"✅ Loaded {file} for {person_name}")

    return db


def load_or_build_db():
    current_sig = get_db_signature(DB_PATH)

    if os.path.exists(CACHE_FILE) and os.path.exists(SIGNATURE_FILE):
        with open(SIGNATURE_FILE, "r") as f:
            if f.read() == current_sig:
                print("⚡ Loading embeddings from cache...")
                return np.load(CACHE_FILE, allow_pickle=True).item()

    print("🔄 Rebuilding face database...")
    db = load_face_database(DB_PATH)
    np.save(CACHE_FILE, db)

    with open(SIGNATURE_FILE, "w") as f:
        f.write(current_sig)

    return db


known_faces = load_or_build_db()
db_lock = threading.Lock()


def add_face(name, embedding):
    with db_lock:
        if name not in known_faces:
            known_faces[name] = []
        known_faces[name].append(embedding)
        np.save(CACHE_FILE, known_faces)
        with open(SIGNATURE_FILE, "w") as f:
            f.write(get_db_signature(DB_PATH))
    print(f"✅ Saved face for {name} (total: {len(known_faces[name])})")


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def recognize_face(embedding, threshold=0.5):
    best_match = "Unknown"
    best_score = 0

    if embedding is None:
        return best_match

    with db_lock:
        for name, embeddings in known_faces.items():
            for ref_emb in embeddings:
                score = cosine_similarity(embedding, ref_emb)
                if score > best_score:
                    best_score = score
                    best_match = name

    return best_match if best_score > threshold else "Unknown"


def is_good_embedding(embedding):
    return embedding is not None and np.linalg.norm(embedding) > 0.5


class VideoPanel(QWidget):
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
        pixmap = QPixmap.fromImage(image).scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(pixmap)

    def set_status(self, status):
        self.status_label.setText(status)


class VideoWorker(QThread):
    frame_ready = pyqtSignal(str, object, object)
    status = pyqtSignal(str, str)

    def __init__(self, source_name, stream_url):
        super().__init__()
        self.source_name = source_name
        self.stream_url = stream_url
        self._running = False
        self.track_identities = {}

    def run(self):
        cap = cv2.VideoCapture(self.stream_url)
        if not cap.isOpened():
            self.status.emit(self.source_name, "Unable to open video stream")
            return

        self._running = True

        while self._running:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            results = analyzer.analyze(frame)
            frame = self.process_faces(frame, results)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
            face_data = [
                {
                    "track_id": getattr(face, "track_id", None),
                    "embedding": getattr(face, "embedding", None),
                    "bbox": tuple(map(int, face.bbox)),
                }
                for face in results
            ]
            self.frame_ready.emit(self.source_name, qimg, face_data)

        cap.release()

    def stop(self):
        self._running = False
        self.wait()

    def process_faces(self, frame, results):
        now = time.time()
        self.track_identities = {
            tid: data
            for tid, data in self.track_identities.items()
            if now - data["last_seen"] < MAX_AGE
        }

        for face in results:
            embedding = getattr(face, "embedding", None)
            track_id = getattr(face, "track_id", None)
            name = "Unknown"

            if track_id is not None:
                if track_id in self.track_identities:
                    name = self.track_identities[track_id]["name"]
                    if is_good_embedding(embedding):
                        new_name = recognize_face(embedding)
                        if new_name != "Unknown":
                            self.track_identities[track_id]["name"] = new_name
                            name = new_name
                else:
                    if is_good_embedding(embedding):
                        name = recognize_face(embedding)
                    self.track_identities[track_id] = {
                        "name": name,
                        "last_seen": now,
                    }
                self.track_identities[track_id]["last_seen"] = now
            else:
                if is_good_embedding(embedding):
                    name = recognize_face(embedding)

            self.draw_overlay(frame, face, name)

        return frame

    @staticmethod
    def draw_overlay(frame, face, name):
        x1, y1, x2, y2 = tuple(map(int, face.bbox))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if getattr(face, "track_id", None) is not None:
            cv2.putText(
                frame,
                f"ID:{getattr(face, 'track_id', '')}",
                (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                2,
            )


class LANScanner(QThread):
    scan_result = pyqtSignal(list)
    status = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._running = False

    def run(self):
        self._running = True
        local_ip = self.get_local_ip()
        if local_ip is None:
            self.status.emit("Unable to determine local LAN IP")
            return

        base = ".".join(local_ip.split(".")[:3])
        self.status.emit(f"Scanning LAN {base}.x ...")

        last_octet = int(local_ip.split(".")[-1])
        start = max(1, last_octet - 20)
        end = min(254, last_octet + 20)
        candidates = [f"{base}.{i}" for i in range(start, end + 1) if i != last_octet]

        found_sources = []
        for ip in candidates:
            if not self._running:
                break
            if not self.host_has_open_port(ip, [80, 8080, 554, 8554]):
                continue

            for url in self.common_stream_urls(ip):
                if not self._running:
                    break
                if self.try_open_stream(url):
                    found_sources.append((f"Camera {len(found_sources) + 1}", url))
                    break

        if found_sources:
            self.scan_result.emit(found_sources)
            self.status.emit(f"Detected {len(found_sources)} camera(s)")
        else:
            self.status.emit("No cameras found")

    def get_local_ip(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.connect(("8.8.8.8", 80))
            local_ip = sock.getsockname()[0]
            sock.close()
            return local_ip
        except Exception:
            return None

    def host_has_open_port(self, host, ports):
        for port in ports:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.12)
                sock.connect((host, port))
                sock.close()
                return True
            except Exception:
                continue
        return False

    def common_stream_urls(self, ip):
        return [
            f"http://{ip}/video",
            f"http://{ip}:8080/video",
            f"http://{ip}:81/video",
            f"http://{ip}:80/video",
            f"rtsp://{ip}/stream",
            f"rtsp://{ip}:554/stream",
            f"rtsp://{ip}:8554/live",
            f"rtsp://{ip}:8554/0",
        ]

    def try_open_stream(self, url):
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            cap.release()
            return False

        ret, _ = cap.read()
        cap.release()
        return bool(ret)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UniFace Live")
        self.setMinimumSize(1200, 780)

        self.sources = {}
        self.last_face_data = {}

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

        self.auto_detect_button = QPushButton("Auto-Detect Cameras")
        self.auto_detect_button.clicked.connect(self.auto_detect_sources)

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

        scan_action_layout = QHBoxLayout()
        scan_action_layout.addWidget(self.auto_detect_button)
        scan_action_layout.addWidget(self.refresh_button)

        left_panel = QVBoxLayout()
        left_panel.addWidget(QLabel("Sources"))
        left_panel.addWidget(self.source_list, stretch=1)
        left_panel.addLayout(source_controls_layout)
        left_panel.addLayout(source_action_layout)
        left_panel.addLayout(scan_action_layout)
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
        source = self.sources.get(name)
        if not source:
            return

        worker = source.get("worker")
        if worker:
            worker.stop()
            source["worker"] = None
            source["panel"].set_status("Stopped")

    def start_all_sources(self):
        for name in self.sources:
            self.start_source(name)

    def stop_all_sources(self):
        for name in self.sources:
            self.stop_source(name)
        self.stop_all_button.setEnabled(False)

    def update_video_layout(self):
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
        global known_faces
        self.status_label.setText("Reloading face database...")
        known_faces = load_or_build_db()
        self.status_label.setText("Face database refreshed")

    def remove_selected_source(self):
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

    def auto_detect_sources(self):
        self.scan_thread = LANScanner()
        self.scan_thread.scan_result.connect(self.on_scan_complete)
        self.scan_thread.status.connect(self.on_scan_status)
        self.scan_thread.finished.connect(lambda: self.auto_detect_button.setEnabled(True))
        self.auto_detect_button.setEnabled(False)
        self.status_label.setText("Scanning LAN for cameras...")
        self.scan_thread.start()

    def on_scan_complete(self, detected_sources):
        for name, url in detected_sources:
            if name in self.sources:
                continue
            panel = VideoPanel(name)
            self.sources[name] = {
                "url": url,
                "worker": None,
                "panel": panel,
            }
            self.source_list.addItem(QListWidgetItem(name))
        self.update_video_layout()
        self.auto_detect_button.setEnabled(True)

    def on_scan_status(self, message):
        self.status_label.setText(message)

    def on_frame_ready(self, source_name, image, face_data):
        source = self.sources.get(source_name)
        if not source:
            return

        source["panel"].update_frame(image)
        self.last_face_data[source_name] = face_data
        source["panel"].set_status(f"Faces: {len(face_data)}")
        self.status_label.setText(f"{source_name}: {len(face_data)} faces")

    def on_status(self, source_name, message):
        source = self.sources.get(source_name)
        if source:
            source["panel"].set_status(message)
        self.status_label.setText(f"{source_name}: {message}")

    def on_source_selected(self, current, previous):
        self.remove_source_button.setEnabled(current is not None)
        if current:
            self.status_label.setText(f"Selected source: {current.text()}")

    def closeEvent(self, event):
        self.stop_all_sources()
        event.accept()


def main():
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()

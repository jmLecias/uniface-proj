import os
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
    QHBoxLayout,
    QLabel,
    QLineEdit,
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


class VideoWorker(QThread):
    frame_ready = pyqtSignal(object, object)
    status = pyqtSignal(str)

    def __init__(self, stream_url):
        super().__init__()
        self.stream_url = stream_url
        self._running = False
        self.track_identities = {}

    def run(self):
        cap = cv2.VideoCapture(self.stream_url)
        if not cap.isOpened():
            self.status.emit("Unable to open video stream")
            return

        self._running = True
        prev_time = time.time()

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
            self.frame_ready.emit(qimg, face_data)

            now = time.time()
            prev_time = now

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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UniFace Live")
        self.setMinimumSize(960, 700)

        self.video_label = QLabel("Waiting for stream...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background: black; color: white;")

        self.status_label = QLabel("Ready")
        self.stream_input = QLineEdit("http://192.168.254.100:8080/video")

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Person name for new face")

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_stream)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_stream)
        self.stop_button.setEnabled(False)

        self.refresh_button = QPushButton("Refresh DB")
        self.refresh_button.clicked.connect(self.reload_database)

        self.add_face_button = QPushButton("Add Face")
        self.add_face_button.clicked.connect(self.add_face)

        self.last_face_data = []
        self.worker = None

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Stream URL:"))
        controls_layout.addWidget(self.stream_input)
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)

        action_layout = QHBoxLayout()
        action_layout.addWidget(self.name_input)
        action_layout.addWidget(self.add_face_button)
        action_layout.addWidget(self.refresh_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(controls_layout)
        main_layout.addWidget(self.video_label, stretch=1)
        main_layout.addLayout(action_layout)
        main_layout.addWidget(self.status_label)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def start_stream(self):
        if self.worker and self.worker.isRunning():
            return

        self.worker = VideoWorker(self.stream_input.text().strip())
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.status.connect(self.on_status)
        self.worker.start()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Starting stream...")

    def stop_stream(self):
        if self.worker:
            self.worker.stop()
            self.worker = None

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Stopped")

    def reload_database(self):
        global known_faces
        self.status_label.setText("Reloading face database...")
        known_faces = load_or_build_db()
        self.status_label.setText("Face database refreshed")

    def add_face(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing name", "Enter a name before adding a face.")
            return

        if len(self.last_face_data) == 0:
            QMessageBox.warning(self, "No face", "No face detected in the current frame.")
            return

        embedding = self.last_face_data[0].get("embedding")
        if not is_good_embedding(embedding):
            QMessageBox.warning(self, "Invalid embedding", "Unable to extract a good face embedding.")
            return

        add_face(name, embedding)
        self.status_label.setText(f"Added face for {name}")

    def on_frame_ready(self, image, face_data):
        self.last_face_data = face_data
        self.video_label.setPixmap(QPixmap.fromImage(image).scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
        ))
        self.status_label.setText(f"Faces: {len(face_data)}")

    def on_status(self, message):
        self.status_label.setText(message)

    def closeEvent(self, event):
        self.stop_stream()
        event.accept()


def main():
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()

"""Video worker thread for processing video streams."""

import time

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

from face_db import detector, recognizer, tracker, recognize_face, is_good_embedding


class VideoWorker(QThread):
    """Worker thread for capturing and processing video streams."""

    frame_ready = pyqtSignal(str, object, object)
    status = pyqtSignal(str, str)

    def __init__(self, source_name, stream_url):
        super().__init__()
        self.source_name = source_name
        self.stream_url = stream_url
        self._running = False

    def run(self):
        """Main loop for capturing and processing frames."""
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

            faces = detector.detect(frame)
            dets = np.array([[*f.bbox, f.confidence] for f in faces]) if faces else np.empty((0, 5))
            tracks = tracker.update(dets)
            frame = self.process_faces(frame, faces, tracks)

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
                for face in faces
            ]
            self.frame_ready.emit(self.source_name, qimg, face_data)

        cap.release()

    def stop(self):
        """Stop the video processing thread."""
        self._running = False
        self.wait()

    def process_faces(self, frame, faces, tracks):
        """Process detected faces: recognize and draw overlays."""
        for face in faces:
            embedding = None
            if face.landmarks is not None:
                embedding = recognizer.get_normalized_embedding(frame, face.landmarks)
            name = "Unknown"
            if is_good_embedding(embedding):
                name = recognize_face(embedding)

            # Assign track_id from ByteTrack
            if len(tracks) > 0:
                face_centers = np.array([f.bbox[:2] + np.array(f.bbox[2:]) / 2 for f in faces])
                track_centers = tracks[:, :2] + tracks[:, 2:4] / 2
                for ti, track in enumerate(tracks):
                    dists = np.sum((track_centers[ti] - face_centers) ** 2, axis=1)
                    closest_idx = np.argmin(dists)
                    faces[closest_idx].track_id = int(track[4])

            self.draw_overlay(frame, face, name)

        return frame

    @staticmethod
    def draw_overlay(frame, face, name):
        """Draw bounding boxes and labels on the frame."""
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

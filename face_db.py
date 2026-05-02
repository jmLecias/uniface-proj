"""Face database management and recognition functionality."""

import os
import hashlib
import threading

import cv2
import numpy as np
from uniface.detection import RetinaFace
from uniface.recognition import ArcFace
from uniface.tracking import BYTETracker
from uniface import compute_similarity


# Constants
DB_PATH = "face-db"
CACHE_FILE = os.path.join(DB_PATH, "face_cache.npy")
SIGNATURE_FILE = os.path.join(DB_PATH, "face_cache_sig.txt")

# Initialize models
detector = RetinaFace()
recognizer = ArcFace()
tracker = BYTETracker(track_thresh=0.5, track_buffer=30)

# Global face database
known_faces = {}
db_lock = threading.Lock()


def get_db_signature(path=DB_PATH):
    """Generate a signature hash of the face database based on file timestamps."""
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
    """Load face embeddings from the database directory."""
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
            faces = detector.detect(img)

            if len(faces) == 0:
                print(f"⚠️ No face found in {img_path}")
                continue

            face = faces[0]
            if face.landmarks is not None:
                embedding = recognizer.get_normalized_embedding(img, face.landmarks)
                db[person_name].append(embedding)
                print(f"✅ Loaded {file} for {person_name}")

    return db


def load_or_build_db():
    """Load face database from cache or rebuild if outdated."""
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


def recognize_face(embedding, threshold=0.5):
    """Recognize a face by comparing its embedding to known faces."""
    best_match = "Unknown"
    best_score = 0

    if embedding is None:
        return best_match

    with db_lock:
        for name, embeddings in known_faces.items():
            for ref_emb in embeddings:
                score = compute_similarity(embedding, ref_emb, normalized=True)
                if score > best_score:
                    best_score = score
                    best_match = name

    return best_match if best_score > threshold else "Unknown"


def is_good_embedding(embedding):
    """Check if an embedding is valid for recognition."""
    return embedding is not None and np.linalg.norm(embedding) > 0.5


def reload_database():
    """Reload the face database from disk."""
    global known_faces
    known_faces = load_or_build_db()
    return known_faces


# Initialize the database on import
known_faces = load_or_build_db()

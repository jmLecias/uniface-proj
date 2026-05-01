import cv2
from uniface import FaceAnalyzer
import time
import numpy as np
import os
import hashlib

MAX_AGE = 2.0  # seconds before forgetting a track

# 🔥 Try GPU
try:
    analyzer = FaceAnalyzer(providers=["CUDAExecutionProvider"])
except TypeError:
    analyzer = FaceAnalyzer()


# =========================
# DATABASE LOADING
# =========================

def get_db_signature(path="face-db"):
    file_data = []

    for root, _, files in os.walk(path):
        for f in files:
            if f in ["face_cache.npy", "face_cache_sig.txt"]:
                continue

            full_path = os.path.join(root, f)
            mtime = os.path.getmtime(full_path)
            file_data.append(f"{full_path}:{mtime}")

    file_data.sort()
    return hashlib.md5("".join(file_data).encode()).hexdigest()


def load_face_database(path="face-db"):
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


DB_PATH = "face-db"
CACHE_FILE = os.path.join(DB_PATH, "face_cache.npy")
SIGNATURE_FILE = os.path.join(DB_PATH, "face_cache_sig.txt")


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


# =========================
# FACE LOGIC
# =========================

def add_face(name, embedding):
    if name not in known_faces:
        known_faces[name] = []
    known_faces[name].append(embedding)
    print(f"✅ Saved face for {name} (total: {len(known_faces[name])})")


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def recognize_face(embedding, threshold=0.5):
    best_match = "Unknown"
    best_score = 0

    for name, embeddings in known_faces.items():
        for ref_emb in embeddings:
            score = cosine_similarity(embedding, ref_emb)
            if score > best_score:
                best_score = score
                best_match = name

    return best_match if best_score > threshold else "Unknown"


def is_good_embedding(embedding):
    return embedding is not None and np.linalg.norm(embedding) > 0.5


# =========================
# VIDEO STREAM
# =========================

stream_url = "http://192.168.254.100:8080/video"
cap = cv2.VideoCapture(stream_url)

track_identities = {}  # track_id -> {name, last_seen}

prev_time = 0


# =========================
# MAIN LOOP
# =========================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    results = analyzer.analyze(display_frame)

    key = cv2.waitKey(1) & 0xFF

    # =========================
    # TRACK CLEANUP
    # =========================
    now = time.time()
    track_identities = {
        tid: data
        for tid, data in track_identities.items()
        if now - data["last_seen"] < MAX_AGE
    }

    # =========================
    # PROCESS FACES
    # =========================
    for face in results:

        x1, y1, x2, y2 = map(int, face.bbox)

        track_id = getattr(face, "track_id", None)
        embedding = getattr(face, "embedding", None)

        name = "Unknown"

        # =========================
        # TRACKING LOGIC
        # =========================
        if track_id is not None:

            # EXISTING TRACK
            if track_id in track_identities:
                name = track_identities[track_id]["name"]

                # only improve identity if embedding is good
                if is_good_embedding(embedding):
                    new_name = recognize_face(embedding)

                    if new_name != "Unknown":
                        track_identities[track_id]["name"] = new_name
                        name = new_name

            # NEW TRACK
            else:
                if is_good_embedding(embedding):
                    name = recognize_face(embedding)

                track_identities[track_id] = {
                    "name": name,
                    "last_seen": time.time()
                }

            # 🔥 ALWAYS UPDATE LAST SEEN (IMPORTANT FIX)
            track_identities[track_id]["last_seen"] = time.time()

        else:
            # fallback (no BYTETrack)
            if is_good_embedding(embedding):
                name = recognize_face(embedding)

        
        # =========================
        # DRAW
        # =========================
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(display_frame, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)

        if track_id is not None:
            cv2.putText(display_frame, f"ID:{track_id}", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 255), 2)

    # =========================
    # FPS
    # =========================
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 2)

    cv2.imshow("UniFace", display_frame)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
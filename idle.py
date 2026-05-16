import cv2
import numpy as np
import time
import os
import psycopg2
from ultralytics import YOLO
from config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT

# ========================= MODEL =========================
model = YOLO("yolo11n-pose.pt")

# ========================= SETTINGS =========================
IDLE_THRESHOLD_SECONDS = 10
MOTION_SENSITIVITY = 0.005

ZONE_ID = "Zone 1"
SAVE_FOLDER = "idle_snapshots"

os.makedirs(SAVE_FOLDER, exist_ok=True)

# ========================= DATABASE =========================
DB_PARAMS = {
    "host": DB_HOST,
    "database": DB_NAME,
    "user": DB_USER,
    "password": DB_PASSWORD,
    "port": DB_PORT,
    "sslmode": "require"
}

def init_db():
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS inactivity_log (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            zone_id TEXT,
            worker_id TEXT,
            idle_duration_seconds INTEGER,
            image_path TEXT
        );
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("Database Ready")

def log_idle_event(zone, worker_id, duration, image_path):
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO inactivity_log (zone_id, worker_id, idle_duration_seconds, image_path)
        VALUES (%s, %s, %s, %s)
    """, (zone, worker_id, int(duration), image_path))

    conn.commit()
    cur.close()
    conn.close()

# ========================= INIT =========================
init_db()

# ========================= TRACKING =========================
person_history = {}
id_map = {}
next_id = 1

print("System Started...")

# ========================= YOLO TRACK =========================
results = model.track(
    source=1,
    stream=True,
    persist=True,
    tracker="bytetrack.yaml"
)

# ========================= MAIN LOOP =========================
for result in results:

    frame = result.orig_img.copy()
    current_time = time.time()

    if result.boxes.id is None:
        cv2.imshow("Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    boxes = result.boxes.xyxy.cpu().numpy()
    track_ids = result.boxes.id.cpu().numpy().astype(int)

    for box, tid in zip(boxes, track_ids):

        x1, y1, x2, y2 = map(int, box)

        # assign simple display ID
        if tid not in id_map:
            id_map[tid] = next_id
            next_id += 1

        display_id = id_map[tid]

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if tid not in person_history:
            person_history[tid] = {
                "last_pos": (cx, cy),
                "start_time": current_time,
                "idle_logged": False
            }

        last_x, last_y = person_history[tid]["last_pos"]
        dist = np.sqrt((cx - last_x)**2 + (cy - last_y)**2)

        # ================= ACTIVE =================
        if dist > 20:
            person_history[tid]["last_pos"] = (cx, cy)
            person_history[tid]["start_time"] = current_time
            person_history[tid]["idle_logged"] = False

            status = "ACTIVE"
            color = (0, 255, 0)

        # ================= IDLE =================
        else:
            idle_time = current_time - person_history[tid]["start_time"]

            if idle_time > IDLE_THRESHOLD_SECONDS:
                status = f"IDLE {int(idle_time)}s"
                color = (0, 0, 255)

                if not person_history[tid]["idle_logged"]:

                    timestamp = time.strftime("%Y%m%d_%H%M%S")

                    filename = f"worker_{display_id}_{timestamp}.jpg"
                    path = os.path.join(SAVE_FOLDER, filename)

                    crop = frame[y1:y2, x1:x2]
                    cv2.imwrite(path, crop)

                    log_idle_event(
                        ZONE_ID,
                        f"Worker_{display_id}",
                        idle_time,
                        path
                    )

                    person_history[tid]["idle_logged"] = True

            else:
                status = "Watching..."
                color = (0, 255, 255)

        # DRAW
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"ID {display_id}: {status}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    cv2.imshow("AI Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
import cv2
import os
import time
import numpy as np
import face_recognition
import psycopg2
from ultralytics import YOLO
from config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT

# ===========SETTINGS===================
model = YOLO("yolo11n.pt") 

SAVE_FOLDER = "unauthorized_snapshots"
EMPLOYEE_FOLDER = "employee"
ZONE = "Zone A"
CAMERA_ID = "CAM-01"

# To avoid duplicate logs every millisecond
last_log_time = 0
LOG_COOLDOWN = 10 # Seconds to wait before logging the same alert again

os.makedirs(SAVE_FOLDER, exist_ok=True)

# ============DATABASE=============
def get_conn():
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT,
        sslmode="require"
    )

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS unauthorized_access (
            id SERIAL PRIMARY KEY,
            person_name TEXT,
            camera_id TEXT,
            zone TEXT,
            image_path TEXT,
            access_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    cur.close()
    conn.close()

def log_event(name, cam, zone, path, status):
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO unauthorized_access 
            (person_name, camera_id, zone, image_path)
            VALUES (%s, %s, %s, %s)
        """, (name, cam, zone, path))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Database Error: {e}")

# ===================LOAD AUTHORIZED FACES================
print("Loading employee faces...")
known_encodings = []
known_names = []

if not os.path.exists(EMPLOYEE_FOLDER):
    os.makedirs(EMPLOYEE_FOLDER)
    print(f"Please put employee images in the '{EMPLOYEE_FOLDER}' folder.")

for img_name in os.listdir(EMPLOYEE_FOLDER):
    path = os.path.join(EMPLOYEE_FOLDER, img_name)
    try:
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append("Employee")
    except:
        continue

print(f"Loaded {len(known_encodings)} face samples")
init_db()

# ===================START SYSTEM================
cap = cv2.VideoCapture(1) 
print("System Started...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. ONLY detect 'person' - Class 0 in YOLO COCO dataset
    results = model(frame, classes=[0], conf=0.5) 

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Crop the person from the frame
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            # 2. Find faces ONLY within the person crop
            rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_crop)
            face_encodings = face_recognition.face_encodings(rgb_crop, face_locations)

            label = "PERSON" # Default label if face not clearly seen
            color = (255, 255, 0) # Yellow for 'Checking'

            if not face_encodings:
                label = "FACE NOT SEEN"
            
            for encoding in face_encodings:
                matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
                
                if True in matches:
                    label = "AUTHORIZED"
                    color = (0, 255, 0)
                else:
                    label = "UNAUTHORIZED"
                    color = (0, 0, 255)

                    # 3. Log to DB with Cooldown
                    current_time = time.time()
                    if current_time - last_log_time > LOG_COOLDOWN:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"unauth_{timestamp}.jpg"
                        filepath = os.path.join(SAVE_FOLDER, filename)
                        
                        cv2.imwrite(filepath, frame)
                        log_event("Unknown", CAMERA_ID, ZONE, filepath, "ALERT")
                        last_log_time = current_time
                        print("Alert logged to Database.")

            # Draw UI
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Access Control System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
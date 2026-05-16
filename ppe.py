import cv2
import os
import time
import psycopg2

from ultralytics import YOLO

from config import (
    DB_HOST,
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
    DB_PORT
)

# DATABASE FUNCTIONS
def get_connection():

    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT,
        sslmode="require"
    )

def init_db():

    conn = get_connection()

    cur = conn.cursor()

    cur.execute("""

        CREATE TABLE IF NOT EXISTS ppe_violations (

            Sl SERIAL PRIMARY KEY,

            worker_id VARCHAR(50),
                
            violation_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            zone VARCHAR(100),

            missing_ppe TEXT,

            image_path TEXT

        )

    """)

    conn.commit()

    cur.close()

    conn.close()

def log_violation(
    worker_id,
    zone,
    missing_ppe,
    image_path
):

    conn = get_connection()

    cur = conn.cursor()

    cur.execute("""

        INSERT INTO ppe_violations (

            worker_id,
            zone,
            missing_ppe,
            image_path

        )

        VALUES (%s, %s, %s, %s)

    """, (

        worker_id,
        zone,
        missing_ppe,
        image_path

    ))

    conn.commit()

    cur.close()

    conn.close()

# SETTINGS

model = YOLO("best.pt")

PERSON_CLASS = "Person"

REQUIRED_PPE = [
    "helmet",
    "gloves",
    "vest"
]

ZONE_NAME = "Zone A"

SAVE_FOLDER = "ppe_violations"

# CREATE FOLDER
if not os.path.exists(SAVE_FOLDER):

    os.makedirs(SAVE_FOLDER)
    
# INITIALIZE DATABASE
init_db()

# WEBCAM
cap = cv2.VideoCapture(1)

# OVERLAP FUNCTION
def get_overlap_ratio(box_a, box_b):

    xA = max(box_a[0], box_b[0])

    yA = max(box_a[1], box_b[1])

    xB = min(box_a[2], box_b[2])

    yB = min(box_a[3], box_b[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)

    box_a_area = (
        (box_a[2] - box_a[0]) *
        (box_a[3] - box_a[1])
    )

    if box_a_area == 0:

        return 0

    return inter_area / box_a_area

# START
print("PPE Monitoring Started...")

violation_cooldown = {}

# MAIN LOOP
while True:

    ret, frame = cap.read()

    if not ret:

        print("Camera read failed")

        break

    # YOLO Detection
    results = model(frame, conf=0.4)

    persons = []

    ppe_items = []

    # DETECTION PROCESSING
    for r in results:

        for box_data in r.boxes:

            box = box_data.xyxy.cpu().numpy()[0]

            cls_id = int(
                box_data.cls.cpu().numpy()[0]
            )

            class_name = model.names[cls_id]

            x1, y1, x2, y2 = map(int, box)

            # PERSON
            if class_name == PERSON_CLASS:

                persons.append({
                    "box": box
                })

            # PPE
            elif class_name in REQUIRED_PPE:

                ppe_items.append({
                    "box": box,
                    "name": class_name
                })

                # Draw PPE box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    class_name,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )
                
    # PPE COMPLIANCE CHECK
    for idx, person in enumerate(persons):

        person_box = person["box"]

        found_ppe = set()

        for ppe in ppe_items:

            overlap = get_overlap_ratio(
                ppe["box"],
                person_box
            )

            if overlap > 0.1:

                found_ppe.add(
                    ppe["name"]
                )

        compliant = all(
            item in found_ppe
            for item in REQUIRED_PPE
        )

        px1, py1, px2, py2 = map(
            int,
            person_box
        )

        worker_id = f"Worker_{idx+1}"
        
        # SAFE
        if compliant:

            color = (0, 255, 0)

            label = "Wearing PPE"
            
        # VIOLATION
        else:

            color = (0, 0, 255)

            missing = [

                item for item in REQUIRED_PPE

                if item not in found_ppe
            ]

            label = "Missing: " + ", ".join(missing)

            current_time = time.time()

            # Prevent duplicate saving
            if (
                worker_id not in violation_cooldown
                or
                current_time -
                violation_cooldown[worker_id] > 10
            ):

                violation_cooldown[
                    worker_id
                ] = current_time

                # Save image
                timestamp = time.strftime(
                    "%Y%m%d_%H%M%S"
                )

                filename = (
                    f"{worker_id}_{timestamp}.jpg"
                )

                filepath = os.path.join(
                    SAVE_FOLDER,
                    filename
                )

                crop = frame[
                    max(0, py1 - 10):py2 + 10,
                    max(0, px1 - 10):px2 + 10
                ]

                cv2.imwrite(
                    filepath,
                    crop
                )

                # Save in PostgreSQL
                log_violation(
                    worker_id=worker_id,
                    zone=ZONE_NAME,
                    missing_ppe=", ".join(missing),
                    image_path=filepath
                )

                print(
                    f"[ALERT] {worker_id} missing {missing}"
                )

        # DRAW PERSON BOX
        cv2.rectangle(
            frame,
            (px1, py1),
            (px2, py2),
            color,
            3
        )

        cv2.putText(
            frame,
            f"{worker_id}: {label}",
            (px1, py1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    # SHOW OUTPUT
    cv2.imshow(
        "PPE Compliance Monitor",
        frame
    )

    # Exit
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break

# CLEANUP
cap.release()
cv2.destroyAllWindows()

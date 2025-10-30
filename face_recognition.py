import cv2
import dlib
import sqlite3
import numpy as np

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("./Models/shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("./Models/dlib_face_recognition_resnet_model_v1.dat")

DB_FILE = "./Datasets/face_db.sqlite"
conn = sqlite3.connect(DB_FILE)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS faces (
    name TEXT PRIMARY KEY,
    embedding BLOB NOT NULL
)
""")
conn.commit()

database = {}
c.execute("SELECT name, embedding FROM faces")
for name, emb_blob in c.fetchall():
    database[name] = np.frombuffer(emb_blob, dtype=np.float64)

def get_embedding(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(rgb)
    if len(faces) == 0:
        return None
    shape = sp(rgb, faces[0])
    return np.array(facerec.compute_face_descriptor(rgb, shape))

def add_new_face(name):
    cap = cv2.VideoCapture(0)
    if any(name.lower() == db_name.lower() for db_name in database):
        print(f"⚠️ Face for '{name}' already exists in database!")
        return
    print(f"Look at the camera to capture face for '{name}'...")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        cv2.imshow("Add New Face - Press 'c' to capture", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('c'):
            emb = get_embedding(frame)
            if emb is not None:
                database[name] = emb
                c.execute("INSERT INTO faces (name, embedding) VALUES (?, ?)", (name, emb.tobytes()))
                conn.commit()
                print(f"Face for '{name}' added!")
            else:
                print("No face detected. Try again.")
            break
        elif key & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def recognize_faces():
    cap = cv2.VideoCapture(0)
    print("Starting face recognition. Press 'q' to quit.")

    frame_count = 0
    n = 3

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if frame_count % n == 0:
            faces = detector(rgb)
        frame_count += 1

        for face in faces:
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            shape = sp(rgb, face)
            emb = np.array(facerec.compute_face_descriptor(rgb, shape))

            name = "Unknown"
            min_dist = 0.6
            for db_name, db_emb in database.items():
                dist = np.linalg.norm(db_emb - emb)
                if dist < min_dist:
                    min_dist = dist
                    name = db_name

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

while True:
    print("\n===== FACE RECOGNITION MENU =====")
    print("1. Add New Face")
    print("2. Recognize Faces")
    print("3. Quit")

    ch = input("Enter your choice: ").strip()
    if ch == '1':
        name = input("Enter the name of the person: ").strip()
        add_new_face(name)
    elif ch == '2':
        if len(database) == 0:
            print("⚠️ Database empty! Add faces first.")
        else:
            recognize_faces()
    elif ch == '3':
        print("Exiting...👋🏻")
        break
    else:
        print("Invalid choice. Please try again!")
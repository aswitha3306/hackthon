from flask import Flask, render_template, request, redirect, Response, send_from_directory
import mysql.connector
import os
from werkzeug.utils import secure_filename
import face_recognition
import numpy as np
import cv2
from datetime import datetime
from ultralytics import YOLO
import pyttsx3
import threading
import time

app = Flask(__name__)

# =========================
# Upload Folder
# =========================
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# =========================
# Evidence Folder
# =========================
EVIDENCE_FOLDER = "evidence"
os.makedirs(EVIDENCE_FOLDER, exist_ok=True)

# =========================
# Globals
# =========================
suspicion_score = 0
activity_text = "Normal"

# Grace period to avoid initial HIGH RISK
start_time = time.time()
GRACE_SECONDS = 5

# =========================
# Serve Evidence
# =========================
@app.route('/evidence/<filename>')
def evidence_file(filename):
    return send_from_directory(EVIDENCE_FOLDER, filename)

# =========================
# Database
# =========================
db = mysql.connector.connect(
    host="localhost",
    port=3307,
    user="root",
    password="",
    database="proctoring"
)

# =========================
# Load Faces
# =========================
KNOWN_FACE_ENCODINGS = []
KNOWN_FACE_IDS = []

def load_faces():

    KNOWN_FACE_ENCODINGS.clear()
    KNOWN_FACE_IDS.clear()

    for file in os.listdir(app.config['UPLOAD_FOLDER']):

        path = os.path.join(app.config['UPLOAD_FOLDER'], file)

        try:
            img = face_recognition.load_image_file(path)
            enc = face_recognition.face_encodings(img)

            if len(enc) > 0:
                KNOWN_FACE_ENCODINGS.append(enc[0])
                KNOWN_FACE_IDS.append(file.split(".")[0])

        except:
            pass

load_faces()

# =========================
# YOLO
# =========================
model = YOLO("yolov8n.pt")

camera = cv2.VideoCapture(0)

# =========================
# Voice System
# =========================
voice_cooldown = 4
last_voice_time = 0

def speak_async(message):

    global last_voice_time

    if time.time() - last_voice_time < voice_cooldown:
        return

    def run():
        engine = pyttsx3.init()
        engine.say(message)
        engine.runAndWait()

    threading.Thread(target=run).start()
    last_voice_time = time.time()

# =========================
# Get Student Name
# =========================
def get_student_name(reg_no):

    try:
        cursor = db.cursor()
        cursor.execute("SELECT name FROM students WHERE reg_no=%s", (reg_no,))
        result = cursor.fetchone()
        cursor.close()

        return result[0] if result else "Unknown"

    except:
        return "Unknown"

# =========================
# Trigger Alert
# =========================
def trigger_alert(student_id, violation_type, frame):

    name = get_student_name(student_id)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{student_id}_{timestamp}.jpg"

    filepath = os.path.join(EVIDENCE_FOLDER, filename)

    cv2.imwrite(filepath, frame)

    speak_async(f"Alert. {name}. {violation_type}")

# =========================
# Video Streaming
# =========================
def gen_frames():

    global suspicion_score
    global activity_text

    while True:

        success, frame = camera.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        student_id = "Unknown"
        student_name = "Unknown"
        activity_text = "Normal"

        # ======================
        # Grace Period
        # ======================
        scoring_enabled = (time.time() - start_time) > GRACE_SECONDS

        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        # ======================
        # MULTIPLE FACES
        # ======================
        if len(face_locations) > 1:

            activity_text = "Multiple Faces Detected"

            if scoring_enabled:
                suspicion_score += 5

            speak_async("Multiple faces detected")

        # ======================
        # FACE PROCESS
        # ======================
        if len(face_locations) >= 1:

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

                face_center_x = (left + right) / 2
                frame_center = frame.shape[1] / 2

                if abs(face_center_x - frame_center) > frame.shape[1] * 0.25:

                    activity_text = "Looking Away"

                    if scoring_enabled:
                        suspicion_score += 2

                    speak_async("Please look at the screen")

                if len(KNOWN_FACE_ENCODINGS) > 0:

                    matches = face_recognition.compare_faces(KNOWN_FACE_ENCODINGS, face_encoding)
                    distances = face_recognition.face_distance(KNOWN_FACE_ENCODINGS, face_encoding)
                    best = np.argmin(distances)

                    if matches[best]:
                        student_id = KNOWN_FACE_IDS[best]
                        student_name = get_student_name(student_id)
                    else:
                        activity_text = "Unknown Person"

                        if scoring_enabled:
                            suspicion_score += 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, student_name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # ======================
        # NO FACE
        # ======================
        if len(face_locations) == 0:

            activity_text = "No Face Detected"

            if scoring_enabled:
                suspicion_score += 2

            speak_async("Face not visible")

        # ======================
        # OBJECT DETECTION
        # ======================
        results = model(frame, verbose=False)

        for r in results:
            for box in r.boxes:

                label = model.names[int(box.cls[0])]

                if label == "cell phone":

                    activity_text = "Mobile Phone Detected"

                    if scoring_enabled:
                        suspicion_score += 5

                    trigger_alert(student_id, "Mobile Phone Detected", frame)

                if label in ["book", "notebook", "paper", "document"]:

                    activity_text = "Bit Paper Detected"

                    if scoring_enabled:
                        suspicion_score += 4

                    trigger_alert(student_id, "Bit Paper Detected", frame)

        # ======================
        # RISK LEVEL
        # ======================
        if suspicion_score < 5:
            risk_text = "SAFE"
            color = (0, 255, 0)

        elif suspicion_score < 15:
            risk_text = "WARNING"
            color = (0, 165, 255)

        else:
            risk_text = "HIGH RISK"
            color = (0, 0, 255)

        # ======================
        # DISPLAY
        # ======================
        cv2.putText(frame,
                    f"Risk: {risk_text}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    3)

        cv2.putText(frame,
                    f"Activity: {activity_text}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2)

        # Risk decay
        suspicion_score = max(0, suspicion_score - 0.02)

        # Stream
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return render_template("login.html")


@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        if username == "admin" and password == "admin":
            return redirect("/dashboard")

    return render_template("login.html")


@app.route("/dashboard")
def dashboard():

    if os.path.exists(EVIDENCE_FOLDER):
        files = os.listdir(EVIDENCE_FOLDER)
        files.sort(reverse=True)
    else:
        files = []

    return render_template("dashboard.html", evidence_files=files)


@app.route('/students')
def students():

    cursor = db.cursor()
    cursor.execute("SELECT * FROM students")
    data = cursor.fetchall()
    cursor.close()

    return render_template('students.html', students=data)


@app.route('/register', methods=['GET', 'POST'])
def register():

    if request.method == 'POST':

        reg_no = request.form['reg_no']
        name = request.form['name']
        department = request.form['department']
        email = request.form['email']
        password = request.form['password']
        photo = request.files['photo']

        photo_filename = secure_filename(photo.filename)
        photo.save(os.path.join(app.config['UPLOAD_FOLDER'], photo_filename))

        cursor = db.cursor()
        cursor.execute("""
            INSERT INTO students (reg_no, name, department, email, password, photo)
            VALUES (%s,%s,%s,%s,%s,%s)
        """, (reg_no, name, department, email, password, photo_filename))

        db.commit()
        cursor.close()

        load_faces()

        return redirect('/students')

    return render_template('register.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=False)
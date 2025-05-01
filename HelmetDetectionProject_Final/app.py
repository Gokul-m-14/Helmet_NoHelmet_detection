from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
import cv2
import geocoder
import firebase_admin
from firebase_admin import credentials, db, storage
from datetime import datetime
import os
import uuid
import threading

app = Flask(__name__)
model = YOLO("E:/Pycharm/YOLOHelmet/runs/detect/train6/weights/best.pt")  # Use your actual model path

cred = credentials.Certificate("firebase_config.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://no-helmet-detection-default-rtdb.firebaseio.com/',
    'storageBucket': 'no-helmet-detection.appspot.com'
})

alert_data = {"alert": False, "location": ""}
is_detecting = {"active": False}
lock = threading.Lock()

def get_location():
    try:
        g = geocoder.ip('me')
        return f"{g.latlng[0]}, {g.latlng[1]}" if g.ok else "Unknown"
    except:
        return "Unknown"

def store_alert_with_image(location, image_path):
    try:
        bucket = storage.bucket()
        image_id = str(uuid.uuid4())
        blob = bucket.blob(f'alerts/{image_id}.jpg')
        blob.upload_from_filename(image_path)
        blob.make_public()
        image_url = blob.public_url

        ref = db.reference('helmet_alerts')
        ref.push({
            "timestamp": datetime.utcnow().isoformat(),
            "location": location,
            "image_url": image_url
        })

        print(f"[INFO] Stored alert with image: {image_url}")
    except Exception as e:
        print("[ERROR] Firebase upload failed:", e)

def detect_objects(frame):
    results = model(frame)
    annotated_frame = results[0].plot()

    names = results[0].names
    classes = results[0].boxes.cls.cpu().tolist()
    helmet_detected = any(names[int(cls)] == "helmet" for cls in classes)

    if not helmet_detected:
        with lock:
            if not alert_data["alert"]:
                alert_data["alert"] = True
                location = get_location()
                alert_data["location"] = location

                image_path = f"no_helmet_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.jpg"
                cv2.imwrite(image_path, frame)
                store_alert_with_image(location, image_path)
                os.remove(image_path)

        cv2.putText(annotated_frame, f"⚠️ No Helmet Detected! Location: {alert_data['location']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        with lock:
            alert_data["alert"] = False
            alert_data["location"] = ""
        cv2.putText(annotated_frame, "✅ Helmet Detected",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return annotated_frame

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        if is_detecting["active"]:
            frame = detect_objects(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_alert')
def check_alert():
    with lock:
        return jsonify(alert_data)

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    action = request.json.get("action")
    if action == "start":
        is_detecting["active"] = True
        return jsonify({"status": "Detection started"})
    elif action == "stop":
        is_detecting["active"] = False
        return jsonify({"status": "Detection stopped"})
    return jsonify({"status": "Invalid action"})

if __name__ == "__main__":
    app.run(debug=True)

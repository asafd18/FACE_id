from flask import Flask, render_template, request, session, redirect, url_for, send_file, Response
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import face_recognition
from PIL import Image, ImageDraw
import torch
import time
from mediapipe import solutions as mp_solutions
import pickle
from pyzbar.pyzbar import decode
import base64
import threading

app = Flask(__name__)
app.secret_key = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = './Uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

socketio = SocketIO(app)
db = SQLAlchemy(app)

# Variables for QR scanner
latest_qr_data = "המתן לקריאת קוד QR..."
capture_active = True

# Create Uploads directory if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Global variables for controlling face detection, object detection, and sign language detection
video_capture = None
running_face = False
running_object = False
running_sign_language = False
model_yolo = None

# MediaPipe Hands setup for sign language detection
mp_hands = mp_solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp_solutions.drawing_utils

# Load sign language model and scaler
try:
    with open('./sign_language_model.pkl', 'rb') as f:
        sign_language_model = pickle.load(f)
    with open('./sign_language_scaler.pkl', 'rb') as f:
        sign_language_scaler = pickle.load(f)
except FileNotFoundError:
    print("Warning: Sign language model or scaler not found. Please train the model first.")

# Database model
class UserData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    data = db.Column(db.Text, nullable=True)

# Create database
with app.app_context():
    db.create_all()

# Load YOLOv5 model
def load_yolo_model():
    global model_yolo
    if model_yolo is None:
        model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model_yolo.conf = 0.1  # Low confidence threshold
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_yolo = model_yolo.to(device)
        if device == 'cuda':
            model_yolo = model_yolo.half()  # FP16 mode for acceleration
        print("YOLOv5 model loaded successfully.")
    return model_yolo

# Check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Convert image to sketch
def convert_to_sketch(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted_img = cv2.bitwise_not(gray_img)
    blurred_img = cv2.GaussianBlur(inverted_img, (21, 21), sigmaX=0, sigmaY=0)
    inverted_blur_img = cv2.bitwise_not(blurred_img)
    sketch_img = cv2.divide(gray_img, inverted_blur_img, scale=256.0)
    return sketch_img

# QR code video capture
def capture_video():
    global latest_qr_data, capture_active
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        latest_qr_data = "שגיאה: לא ניתן לגשת למצלמה. ודא שהמצלמה מחוברת והרשאות ניתנו."
        socketio.emit('qr_update', {'data': latest_qr_data})
        return

    while capture_active:
        ret, frame = cap.read()
        if not ret:
            latest_qr_data = "שגיאה: לא ניתן לקרוא וידאו"
            socketio.emit('qr_update', {'data': latest_qr_data})
            break

        qr_codes = decode(frame)
        if qr_codes:
            qr_data = qr_codes[0].data.decode('utf-8')
            if qr_data != latest_qr_data:
                latest_qr_data = qr_data
                socketio.emit('qr_update', {'data': qr_data})
                capture_active = False
                break

        _, buffer = cv2.imencode('.jpg', frame)
        frame_data = base64.b64encode(buffer).decode('utf-8')
        socketio.emit('video_frame', {'image': frame_data})

    cap.release()
    socketio.emit('video_stopped', {'message': 'המצלמה כבויה'})

# Generate frames for face detection
def generate_frames():
    global video_capture, running_face

    if video_capture is not None:
        video_capture.release()
        video_capture = None
        print("Previous video capture released (face detection).")

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open webcam (face detection).")
        return
    print("Webcam opened successfully (face detection).")

    running_face = True

    try:
        while running_face:
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Could not read frame from webcam (face detection).")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

            pil_image = Image.fromarray(rgb_frame)
            d = ImageDraw.Draw(pil_image)

            for face_landmarks in face_landmarks_list:
                for facial_feature in face_landmarks.keys():
                    d.line(face_landmarks[facial_feature], fill='red', width=5)

            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            ret, buffer = cv2.imencode('.jpg', opencv_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ret:
                print("Error: Could not encode frame to JPEG (face detection).")
                continue
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except Exception as e:
        print(f"Error during face detection streaming: {e}")

    finally:
        if video_capture is not None:
            video_capture.release()
            video_capture = None
            print("Video capture released in finally block (face detection).")
            time.sleep(0.5)
        running_face = False
        print("Face detection stream stopped and resources released.")

# Generate frames for object detection
def generate_object_frames():
    global video_capture, running_object, model_yolo

    if video_capture is not None:
        video_capture.release()
        video_capture = None
        print("Previous video capture released (object detection).")

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open webcam (object detection).")
        return
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Webcam opened successfully (object detection).")

    model_yolo = load_yolo_model()

    running_object = True

    try:
        while running_object:
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Could not read frame from webcam (object detection).")
                break

            frame_resized = cv2.resize(frame, (640, 480))
            results = model_yolo(frame_resized)
            frame_rendered = results.render()[0]

            ret, buffer = cv2.imencode('.jpg', frame_rendered, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ret:
                print("Error: Could not encode frame to JPEG (object detection).")
                continue
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except Exception as e:
        print(f"Error during object detection streaming: {e}")

    finally:
        if video_capture is not None:
            video_capture.release()
            video_capture = None
            print("Video capture released in finally block (object detection).")
            time.sleep(0.5)
        running_object = False
        print("Object detection stream stopped and resources released.")

# Extract hand landmarks for sign language detection
def extract_hand_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
    return None

# Generate frames for sign language detection
def generate_sign_language_frames():
    global video_capture, running_sign_language
    if video_capture is not None:
        video_capture.release()
        video_capture = None
        print("Previous video capture released (sign language detection).")

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open webcam (sign language detection).")
        return
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Webcam opened successfully (sign language detection).")

    running_sign_language = True

    try:
        while running_sign_language:
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Could not read frame from webcam (sign language detection).")
                break

            frame_resized = cv2.resize(frame, (640, 480))
            landmarks = extract_hand_landmarks(frame_resized)

            if landmarks is not None:
                landmarks_scaled = sign_language_scaler.transform([landmarks])
                prediction = sign_language_model.predict(landmarks_scaled)[0]
                confidence = sign_language_model.predict_proba(landmarks_scaled)[0].max()

                cv2.putText(frame_resized, f"Letter: {prediction} ({confidence:.2f})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                results = hands.process(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame_resized, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            ret, buffer = cv2.imencode('.jpg', frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ret:
                print("Error: Could not encode frame to JPEG (sign language detection).")
                continue
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except Exception as e:
        print(f"Error during sign language detection streaming: {e}")

    finally:
        if video_capture is not None:
            video_capture.release()
            video_capture = None
            print("Video capture released in finally block (sign language detection).")
            time.sleep(0.5)
        running_sign_language = False
        print("Sign language detection stream stopped and resources released.")

# Route for QR scanner page
@app.route('/qr_scanner/<username>')
def qr_scanner(username):
    if 'username' not in session or session['username'] != username:
        return "Unauthorized", 401
    global latest_qr_data, capture_active
    capture_active = True
    latest_qr_data = "המתן לקריאת קוד QR..."  # Reset QR data on new scan
    return render_template('index.html')

# SocketIO connect event for QR scanner
@socketio.on('connect')
def handle_connect():
    global latest_qr_data
    emit('qr_update', {'data': latest_qr_data})
    if capture_active:
        threading.Thread(target=capture_video, daemon=True).start()

# SocketIO event to handle stop capture
@socketio.on('stop_capture')
def handle_stop_capture():
    global capture_active
    capture_active = False

# Login route
@app.route('/login/<username>')
def login(username):
    session['username'] = username
    return redirect(url_for('welcome', username=username))

# Welcome page with image processing
@app.route('/welcome/<username>', methods=['GET', 'POST'])
def welcome(username):
    if 'username' not in session or session['username'] != username:
        return "Unauthorized", 401

    saved_data = ""
    if request.method == 'POST' and 'user_input' in request.form:
        data = request.form.get('user_input')
        user_data = UserData.query.filter_by(username=username).first()
        if user_data:
            user_data.data = data
        else:
            user_data = UserData(username=username, data=data)
            db.session.add(user_data)
        db.session.commit()
        saved_data = data

    sketch_image = None
    if request.method == 'POST' and 'image' in request.files:
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = cv2.imread(filepath)
            sketch_img = convert_to_sketch(img)
            sketch_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'sketch_{filename}')
            cv2.imwrite(sketch_filepath, sketch_img)
            sketch_image = f'sketch_{filename}'

    user_data = UserData.query.filter_by(username=username).first()
    saved_data = user_data.data if user_data else saved_data

    profile_image = f"{username}.jpg"
    profile_image_path = os.path.join('./db', profile_image)
    if not os.path.exists(profile_image_path):
        profile_image = None

    return render_template('welcome.html', username=username, saved_data=saved_data, sketch_image=sketch_image,
                           profile_image=profile_image)

# User page route
@app.route('/user_page/<username>')
def user_page(username):
    if 'username' not in session or session['username'] != username:
        return "Unauthorized", 401

    profile_image = f"{username}.jpg"
    profile_image_path = os.path.join('./db', profile_image)
    if not os.path.exists(profile_image_path):
        profile_image = None

    return render_template('user_page.html', username=username, profile_image=profile_image)

# Video feed route for face detection
@app.route('/video_feed/<username>')
def video_feed(username):
    if 'username' not in session or session['username'] != username:
        return "Unauthorized", 401
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Stop face detection stream
@app.route('/stop_video_feed/<username>')
def stop_video_feed(username):
    global running_face, video_capture
    if 'username' not in session or session['username'] != username:
        return "Unauthorized", 401
    running_face = False
    if video_capture is not None:
        video_capture.release()
        video_capture = None
        print("Video capture released in stop_video_feed (face detection).")
        time.sleep(0.5)
    return "Face detection stream stopped."

# Video feed route for object detection
@app.route('/object_feed/<username>')
def object_feed(username):
    if 'username' not in session or session['username'] != username:
        return "Unauthorized", 401
    return Response(generate_object_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Stop object detection stream
@app.route('/stop_object_feed/<username>')
def stop_object_feed(username):
    global running_object, video_capture
    if 'username' not in session or session['username'] != username:
        return "Unauthorized", 401
    running_object = False
    if video_capture is not None:
        video_capture.release()
        video_capture = None
        print("Video capture released in stop_object_feed (object detection).")
        time.sleep(0.5)
    return "Object detection stream stopped."

# Video feed route for sign language detection
@app.route('/sign_language_feed/<username>')
def sign_language_feed(username):
    if 'username' not in session or session['username'] != username:
        return "Unauthorized", 401
    return Response(generate_sign_language_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Stop sign language detection stream
@app.route('/stop_sign_language_feed/<username>')
def stop_sign_language_feed(username):
    global running_sign_language, video_capture
    if 'username' not in session or session['username'] != username:
        return "Unauthorized", 401
    running_sign_language = False
    if video_capture is not None:
        video_capture.release()
        video_capture = None
        print("Video capture released in stop_sign_language_feed.")
        time.sleep(0.5)
    return "Sign language detection stream stopped."

# Serve profile or uploaded images
@app.route('/db/<filename>')
def profile_file(filename):
    return send_file(os.path.join('./db', filename))

@app.route('/Uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
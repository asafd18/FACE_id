from flask import Flask, render_template, request, session, redirect, url_for, send_file, Response
from flask_sqlalchemy import SQLAlchemy
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import face_recognition
from PIL import Image, ImageDraw
import torch
import time

app = Flask(__name__)
app.secret_key = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = './Uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

db = SQLAlchemy(app)

# יצירת תיקיית Uploads אם לא קיימת
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# משתנים גלובליים לשליטה על זיהוי הפנים וזיהוי האובייקטים
video_capture = None
running_face = False
running_object = False
model_yolo = None

# מודל למסד הנתונים
class UserData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    data = db.Column(db.Text, nullable=True)

# יצירת מסד הנתונים
with app.app_context():
    db.create_all()

# טעינת מודל YOLOv5
def load_yolo_model():
    global model_yolo
    if model_yolo is None:
        model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model_yolo.conf = 0.1  # סף ביטחון נמוך
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_yolo = model_yolo.to(device)
        if device == 'cuda':
            model_yolo = model_yolo.half()  # מצב FP16 להאצה
        print("YOLOv5 model loaded successfully.")
    return model_yolo

# פונקציה לבדיקת סיומת קובץ
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# פונקציה להמרת תמונה לסקיצה
def convert_to_sketch(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted_img = cv2.bitwise_not(gray_img)
    blurred_img = cv2.GaussianBlur(inverted_img, (21, 21), sigmaX=0, sigmaY=0)
    inverted_blur_img = cv2.bitwise_not(blurred_img)
    sketch_img = cv2.divide(gray_img, inverted_blur_img, scale=256.0)
    return sketch_img

# פונקציה לעיבוד והזרמת וידאו עבור זיהוי פנים
def generate_frames():
    global video_capture, running_face

    # וידוא שהמצלמה משוחררת לפני הפתיחה
    if video_capture is not None:
        video_capture.release()
        video_capture = None
        print("Previous video capture released (face detection).")

    # פתיחת המצלמה
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

            # המרת הפריים לפורמט JPEG
            ret, buffer = cv2.imencode('.jpg', opencv_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ret:
                print("Error: Could not encode frame to JPEG (face detection).")
                continue
            frame = buffer.tobytes()

            # שליחת הפריים כחלק מזרם MJPEG
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

# פונקציה לעיבוד והזרמת וידאו עבור זיהוי אובייקטים
def generate_object_frames():
    global video_capture, running_object, model_yolo

    # וידוא שהמצלמה משוחררת לפני הפתיחה
    if video_capture is not None:
        video_capture.release()
        video_capture = None
        print("Previous video capture released (object detection).")

    # פתיחת המצלמה
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open webcam (object detection).")
        return
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Webcam opened successfully (object detection).")

    # טעינת מודל YOLOv5
    model_yolo = load_yolo_model()

    running_object = True

    try:
        while running_object:
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Could not read frame from webcam (object detection).")
                break

            # הקטנת גודל הפריים
            frame_resized = cv2.resize(frame, (640, 480))

            # בצע זיהוי חפצים על כל פריים
            results = model_yolo(frame_resized)

            # שימוש ב-render של YOLOv5 לציור מסגרות ותוויות כברירת מחדל
            frame_rendered = results.render()[0]

            # המרת הפריים לפורמט JPEG עם איכות גבוהה
            ret, buffer = cv2.imencode('.jpg', frame_rendered, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ret:
                print("Error: Could not encode frame to JPEG (object detection).")
                continue
            frame = buffer.tobytes()

            # שליחת הפריים כחלק מזרם MJPEG
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

# דף התחברות
@app.route('/login/<username>')
def login(username):
    session['username'] = username
    return redirect(url_for('welcome', username=username))

# דף מותאם אישית עם עיבוד תמונה
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

    return render_template('welcome.html', username=username, saved_data=saved_data, sketch_image=sketch_image, profile_image=profile_image)

# דף ריק מותאם למשתמש
@app.route('/user_page/<username>')
def user_page(username):
    if 'username' not in session or session['username'] != username:
        return "Unauthorized", 401

    profile_image = f"{username}.jpg"
    profile_image_path = os.path.join('./db', profile_image)
    if not os.path.exists(profile_image_path):
        profile_image = None

    return render_template('user_page.html', username=username, profile_image=profile_image)

# נתיב להזרמת וידאו עבור זיהוי פנים
@app.route('/video_feed/<username>')
def video_feed(username):
    if 'username' not in session or session['username'] != username:
        return "Unauthorized", 401

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# נתיב לעצירת זרם זיהוי פנים
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

# נתיב להזרמת וידאו עבור זיהוי אובייקטים
@app.route('/object_feed/<username>')
def object_feed(username):
    if 'username' not in session or session['username'] != username:
        return "Unauthorized", 401

    return Response(generate_object_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# נתיב לעצירת זרם זיהוי אובייקטים
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

# נתיב להורדת התמונה המעובדת או תמונת פרופיל
@app.route('/db/<filename>')
def profile_file(filename):
    return send_file(os.path.join('./db', filename))

@app.route('/Uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
from flask import Flask, render_template, request, session, redirect, url_for, send_file, Response
from flask_sqlalchemy import SQLAlchemy
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import face_recognition
from PIL import Image, ImageDraw
import threading
import time  # הוספתי עבור עיכוב

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

# משתנה גלובלי לשליטה על זיהוי הפנים
video_capture = None
running = False

# מודל למסד הנתונים
class UserData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    data = db.Column(db.Text, nullable=True)

# יצירת מסד הנתונים
with app.app_context():
    db.create_all()

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

# פונקציה לעיבוד והזרמת הווידאו
def generate_frames():
    global video_capture, running

    # וידוא שהמצלמה משוחררת לפני הפתיחה
    if video_capture is not None:
        video_capture.release()
        video_capture = None
        print("Previous video capture released.")

    # פתיחת המצלמה
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return
    print("Webcam opened successfully.")

    running = True

    try:
        while running:
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
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
            ret, buffer = cv2.imencode('.jpg', opencv_image)
            if not ret:
                print("Error: Could not encode frame to JPEG.")
                continue
            frame = buffer.tobytes()

            # שליחת הפריים כחלק מזרם MJPEG
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except Exception as e:
        print(f"Error during streaming: {e}")

    finally:
        if video_capture is not None:
            video_capture.release()
            video_capture = None
            print("Video capture released in finally block.")
            time.sleep(0.5)  # עיכוב קל כדי לתת למערכת זמן לשחרר את המשאבים
        running = False
        print("Video stream stopped and resources released.")

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

# נתיב להזרמת הווידאו
@app.route('/video_feed/<username>')
def video_feed(username):
    if 'username' not in session or session['username'] != username:
        return "Unauthorized", 401

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# נתיב לעצירת הזרם
@app.route('/stop_video_feed/<username>')
def stop_video_feed(username):
    global running, video_capture
    if 'username' not in session or session['username'] != username:
        return "Unauthorized", 401

    running = False
    if video_capture is not None:
        video_capture.release()
        video_capture = None
        print("Video capture released in stop_video_feed.")
        time.sleep(0.5)  # עיכוב קל כדי לתת למערכת זמן לשחרר את המשאבים
    return "Video stream stopped."

# נתיב להורדת התמונה המעובדת או תמונת פרופיל
@app.route('/db/<filename>')
def profile_file(filename):
    return send_file(os.path.join('./db', filename))

@app.route('/Uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
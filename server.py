from flask import Flask, render_template, request, session, redirect, url_for, send_file
from flask_sqlalchemy import SQLAlchemy
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

db = SQLAlchemy(app)

# יצירת תיקיית Uploads אם לא קיימת
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

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

    # טיפול בשמירת נתונים טקסטואליים
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

    # טיפול בהעלאת תמונה
    sketch_image = None
    if request.method == 'POST' and 'image' in request.files:
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # עיבוד התמונה לסקיצה
            img = cv2.imread(filepath)
            sketch_img = convert_to_sketch(img)
            sketch_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'sketch_{filename}')
            cv2.imwrite(sketch_filepath, sketch_img)
            sketch_image = f'sketch_{filename}'

    # שליפת נתונים קיימים
    user_data = UserData.query.filter_by(username=username).first()
    saved_data = user_data.data if user_data else saved_data

    return render_template('welcome.html', username=username, saved_data=saved_data, sketch_image=sketch_image)

# נתיב להורדת התמונה המעובדת
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
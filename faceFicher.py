import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageDraw

# פתח את מצלמת ברירת המחדל (0)
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    while True:
        # קריאת פריים מהמצלמה
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # המרת התמונה ל-RGB (face_recognition דורש RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # זיהוי תווי פנים
        face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

        print("I found {} face(s) in this frame.".format(len(face_landmarks_list)))

        # המרת המסגרת ל-PIL Image כדי לצייר עליה
        pil_image = Image.fromarray(rgb_frame)
        d = ImageDraw.Draw(pil_image)

        # ציור תווי הפנים
        for face_landmarks in face_landmarks_list:
            for facial_feature in face_landmarks.keys():
                print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))
                d.line(face_landmarks[facial_feature], fill='red', width=5)

        # המרת התמונה חזרה ל-OpenCV format (BGR) להצגה
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # הצגת התמונה עם תווי הפנים
        cv2.imshow('Video', opencv_image)

        # יציאה מהלולאה אם לוחצים על 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    # שחרור משאבים
    video_capture.release()
    cv2.destroyAllWindows()
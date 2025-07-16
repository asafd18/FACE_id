import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageDraw

#Open camara
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    while True:
        #Reading a frame from the camara
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # המרת התמונה ל-RGB (face_recognition דורש RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #Facial recognition
        face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

        print("I found {} face(s) in this frame.".format(len(face_landmarks_list)))

        #convert a frame to PIL Image to tour it
        pil_image = Image.fromarray(rgb_frame)
        d = ImageDraw.Draw(pil_image)

        #Drawing facial features
        for face_landmarks in face_landmarks_list:
            for facial_feature in face_landmarks.keys():
                print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))
                d.line(face_landmarks[facial_feature], fill='red', width=5)

        #Convert image back to (BGR) openCv format to show
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        #Displlaying a picture with facial features
        cv2.imshow('Video', opencv_image)

        #Exit the loop if you press 'P'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    # Resource release
    video_capture.release()
    cv2.destroyAllWindows()
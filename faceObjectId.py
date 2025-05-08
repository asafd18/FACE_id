import cv2
import torch

# טען את מודל YOLOv5 (מודל קל יותר)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.1  # סף ביטחון נמוך

# העבר ל-GPU אם זמין
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
if device == 'cuda':
    model = model.half()  # מצב FP16 להאצה

# פתח את המצלמה
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # רזולוציה נמוכה יותר
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("שגיאה: לא ניתן לפתוח את המצלמה")
    exit()

frame_count = 0
last_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("שגיאה: לא ניתן לקרוא פריים")
        break

    # הקטנת גודל הפריים לפני זיהוי
    frame_resized = cv2.resize(frame, (640, 480))

    frame_count += 1
    if frame_count % 2 == 0:  # עבד כל פריים שני
        # בצע זיהוי חפצים
        results = model(frame_resized)

        # הדפס תוצאות לקונסולה
        for *box, conf, cls in results.xyxy[0]:
            label = f'{model.names[int(cls)]} {conf:.2f}'
            print(label)

        # עדכן את הפריים האחרון
        last_frame = results.render()[0]
    else:
        # השתמש בפריים האחרון אם לא עיבדנו את הפריים הנוכחי
        if last_frame is not None:
            frame = cv2.resize(last_frame, (frame.shape[1], frame.shape[0]))
        else:
            frame = frame_resized

    # הצג את הפריים עם הזיהוי
    cv2.imshow('Object Detection', frame)

    # לחץ על 'q' כדי לצאת
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
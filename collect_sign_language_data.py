import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import os

class DataCollectionApp:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x620+150+100")
        self.main_window.title("Collect Sign Language Data")

        # תווית להזנת אות
        self.letter_label = tk.Label(self.main_window, text="Enter letter (A-Z):", font=("Arial", 18))
        self.letter_label.place(x=850, y=50)
        self.letter_entry = tk.Entry(self.main_window, font=("Arial", 18))
        self.letter_entry.place(x=850, y=80, width=100)

        # כפתור לצילום תמונה
        self.capture_button = tk.Button(
            self.main_window, text="Capture", bg="green", fg="white",
            font=("Arial", 20), command=self.capture_image
        )
        self.capture_button.place(x=850, y=400)

        # תצוגת מצלמה
        self.webcam_label = tk.Label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        self.cap = cv2.VideoCapture(0)
        self.data_dir = './sign_language_data'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()
        if ret:
            self.most_recent_capture = frame
            img_ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.most_recent_capture_pil = Image.fromarray(img_)
            imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
            self.webcam_label.imgtk = imgtk
            self.webcam_label.configure(image=imgtk)
        self.webcam_label.after(20, self.process_webcam)

    def capture_image(self):
        letter = self.letter_entry.get().strip().upper()
        if not letter or not letter.isalpha() or len(letter) != 1:
            messagebox.showerror("Error", "Please enter a valid letter (A-Z).")
            return

        letter_dir = os.path.join(self.data_dir, letter)
        if not os.path.exists(letter_dir):
            os.makedirs(letter_dir)

        # שמירת התמונה
        img_count = len(os.listdir(letter_dir))
        img_path = os.path.join(letter_dir, f"{letter}_{img_count + 1}.jpg")
        cv2.imwrite(img_path, self.most_recent_capture)
        messagebox.showinfo("Success", f"Image saved for letter {letter} as {img_path}")

    def start(self):
        self.main_window.mainloop()

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    app = DataCollectionApp()
    app.start()
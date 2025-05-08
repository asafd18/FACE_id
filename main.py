import tkinter as tk
import webbrowser
import cv2
from PIL import Image, ImageTk
import util  # נדרש קובץ בשם util.py
import os
import subprocess


class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x620+150+100")
        self.main_window.title("Face Recognition Login System")

        # כפתור התחברות
        self.login_button_main_window = util.get_button(
            self.main_window, "Login", 'green', self.login
        )
        self.login_button_main_window.place(x=850, y=400)

        # כפתור הרשמה
        self.register_new_user_button_main_window = util.get_button(
            self.main_window, "Register New User", 'gray',
            self.register_new_user, fg='black'
        )
        self.register_new_user_button_main_window.place(x=850, y=500)

        # תצוגת מצלמה
        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        self.add_webcam(self.webcam_label)

        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

    def add_webcam(self, label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)
        self._label = label
        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()
        if ret:
            self.most_recent_capture_arr = frame
            img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
            self.most_recent_capture_pil = Image.fromarray(img_)
            imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
            self._label.imgtk = imgtk
            self._label.configure(image=imgtk)
        self._label.after(20, self.process_webcam)

    def stop_webcam(self):
        # שחרור המצלמה
        if 'cap' in self.__dict__:
            self.cap.release()
            del self.cap
        # הסתרת תצוגת הווידאו
        self.webcam_label.place_forget()

    def login(self):
        unknown_img_path = './.tmp.jpg'
        cv2.imwrite(unknown_img_path, self.most_recent_capture_arr)

        try:
            output = subprocess.check_output(['face_recognition', self.db_dir, unknown_img_path])
            output_text = output.decode('utf-8').strip()
            print("Recognition output:", output_text)

            if (
                    not output_text or
                    'unknown_person' in output_text.lower() or
                    'no_persons_found' in output_text.lower() or
                    ',' not in output_text
            ):
                self.show_unknown_user_screen()
            else:
                name = output_text.split(',')[1].strip()
                util.msg_box('Welcome back!', f'Welcome, {name}')

                # עצירת המצלמה לאחר זיהוי מוצלח
                self.stop_webcam()

                # פתיחת דף מותאם בשרת המקומי
                webbrowser.open(f"http://127.0.0.1:5000/login/{name}")

                # סגירת החלון הראשי
                self.main_window.destroy()

        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            util.msg_box('Error', f'Face recognition failed: {e}')

    def show_unknown_user_screen(self):
        # הסתרת כל הרכיבים הקיימים
        self.login_button_main_window.place_forget()
        self.register_new_user_button_main_window.place_forget()
        self.webcam_label.place_forget()

        # הצגת ההודעה על המסך
        unknown_user_label = tk.Label(self.main_window, text="Ups...\nUnknown user. Please register new user or try again.",
                                      font=("Arial", 20, "bold"), fg="red")
        unknown_user_label.place(relx=0.5, rely=0.5, anchor="center")

        # כפתור להרשמה חדשה
        try_again_button = util.get_button(self.main_window, "Try Again", "red", self.try_again)
        try_again_button.place(relx=0.5, rely=0.7, anchor="center")

    def try_again(self):
        # החזרת כל הרכיבים הקודמים
        self.login_button_main_window.place(x=850, y=400)
        self.register_new_user_button_main_window.place(x=850, y=500)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        # מחיקת ההודעה והכפתור שנוספו
        for widget in self.main_window.winfo_children():
            if isinstance(widget, tk.Label) and widget.cget("text").startswith("Ups..."):
                widget.destroy()
            if isinstance(widget, tk.Button) and widget.cget("text") == "Try Again":
                widget.destroy()

        # הפעלה מחדש של המצלמה
        self.add_webcam(self.webcam_label)

    def register_new_user(self):
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("1200x620+150+100")
        self.register_new_user_window.title("Register New User")

        self.capture_label = tk.Label(self.register_new_user_window, bg="white")
        self.capture_label.place(x=10, y=0, width=700, height=500)

        self.username_label = tk.Label(
            self.register_new_user_window,
            text="Please input username:",
            font=("Arial", 18, "bold")
        )
        self.username_label.place(x=850, y=70)

        self.username_entry = tk.Entry(
            self.register_new_user_window,
            font=("Arial", 18)
        )
        self.username_entry.place(x=850, y=100, width=250)

        self.accept_button_register_new_user_window = util.get_button(
            self.register_new_user_window, 'Accept', 'green', self.accept_register_new_user
        )
        self.accept_button_register_new_user_window.place(x=850, y=400)

        self.try_again_button_register_new_user_window = util.get_button(
            self.register_new_user_window, 'Try again', 'red', self.try_again_register_new_user
        )
        self.try_again_button_register_new_user_window.place(x=850, y=500)

        self.capture_image()

    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            self.most_recent_capture_arr = frame
            img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
            self.most_recent_capture_pil = Image.fromarray(img_)
            self.register_new_user_capture = self.most_recent_capture_arr
            self.add_img_to_label(self.capture_label, self.most_recent_capture_pil)

    def accept_register_new_user(self):
        name = self.username_entry.get()
        if name:
            cv2.imwrite(os.path.join(self.db_dir, f'{name}.jpg'), self.register_new_user_capture)
            util.msg_box('Success!', 'User was successfully registered!')
            self.register_new_user_window.destroy()
        else:
            util.msg_box('Error', 'Please enter a valid username.')

    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()

    def add_img_to_label(self, label, image):
        imgtk = ImageTk.PhotoImage(image=image)
        label.imgtk = imgtk
        label.configure(image=imgtk)

    def start(self):
        self.main_window.mainloop()


if __name__ == "__main__":
    app = App()
    app.start()
import tkinter as tk
from tkinter import *
import cv2
import csv
import os
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time

# Initialize Haar Cascade face detector
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create main window
window = tk.Tk()
window.title("Face Recognition Attendance System")
window.geometry('1280x720')
window.configure(background='grey80')

# Create necessary directories
os.makedirs("TrainingImage", exist_ok=True)
os.makedirs("TrainingImageLabel", exist_ok=True)
os.makedirs("Attendance", exist_ok=True)
os.makedirs("StudentDetails", exist_ok=True)

# ================== Functions ==================

# Take images for dataset
def take_img():
    enrollment = txt.get()
    name = txt2.get()

    if not enrollment or not name:
        error_msg("Enrollment & Name required!")
        return

    try:
        cam = cv2.VideoCapture(0)
        sampleNum = 0

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                sampleNum += 1
                cv2.imwrite(f"TrainingImage/{name}.{enrollment}.{sampleNum}.jpg", gray[y:y+h, x:x+w])
                cv2.imshow('Capturing Faces', img)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            if sampleNum > 50:
                break

        cam.release()
        cv2.destroyAllWindows()

        with open("StudentDetails/StudentDetails.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([enrollment, name, datetime.datetime.now().strftime("%Y-%m-%d"), datetime.datetime.now().strftime("%H:%M:%S")])

        success_msg(f"Images saved for {name} ({enrollment})")

    except Exception as e:
        error_msg(f"Error: {str(e)}")

# Train the face recognition model
def trainimg():
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        faces, ids = getImagesAndLabels("TrainingImage")
        recognizer.train(faces, np.array(ids))
        recognizer.save("TrainingImageLabel/Trainner.yml")
        success_msg("Model trained successfully!")
    except Exception as e:
        error_msg(f"Training Error: {str(e)}")

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)

    return faceSamples, ids

# Mark attendance with duplicate prevention
def mark_attendance(enrollment, name, subject):
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.datetime.now().strftime("%H:%M:%S")
    filename = f"Attendance/{subject}_{date}.csv"

    already_marked = set()
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0] == str(enrollment):
                    return  # Already marked

    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([enrollment, name, date, time_now])
        print(f"✔ Marked: {enrollment} - {name} for {subject} at {time_now}")

# Automatic attendance
def subjectchoose():
    def fill_attendance():
        subject = tx.get()
        if not subject:
            error_msg("Please enter subject name!")
            return

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("TrainingImageLabel/Trainner.yml")

        cam = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_SIMPLEX

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

                if confidence < 70:
                    student_data = pd.read_csv("StudentDetails/StudentDetails.csv")
                    matched = student_data.loc[student_data['Enrollment'].astype(str) == str(id)]

                    if not matched.empty:
                        student_name = matched['Name'].values[0]
                    else:
                        student_name = "Unknown"
                        print(f"⚠️ Enrollment ID {id} not found.")

                    mark_attendance(id, student_name, subject)
                    cv2.putText(img, f"{id}-{student_name}", (x, y+h), font, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(img, "Unknown", (x, y+h), font, 1, (0, 0, 255), 2)

                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.imshow('Attendance System', img)
            if cv2.waitKey(1) == 27:  # ESC
                break

        cam.release()
        cv2.destroyAllWindows()

    # GUI for subject input
    windo = tk.Toplevel()
    windo.title("Enter Subject")
    windo.geometry('400x200')

    sub = tk.Label(windo, text="Enter Subject:", font=('times', 15))
    sub.pack(pady=10)

    tx = tk.Entry(windo, font=('times', 15))
    tx.pack(pady=10)

    fill_btn = tk.Button(windo, text="Start Attendance", command=fill_attendance, font=('times', 15))
    fill_btn.pack(pady=10)

# Admin Panel
def admin_panel():
    admin_win = tk.Toplevel()
    admin_win.title("Admin Panel")
    admin_win.geometry('800x600')

    try:
        data = pd.read_csv("StudentDetails/StudentDetails.csv")
        for i, row in data.iterrows():
            tk.Label(admin_win, text=f"ID: {row['Enrollment']}, Name: {row['Name']}", font=('times', 12)).pack(pady=5)
    except Exception as e:
        error_msg(f"Error loading data: {str(e)}")

# Notifications
def error_msg(msg):
    Notification.configure(text=msg, bg="red", fg="white")
    Notification.place(x=300, y=400)

def success_msg(msg):
    Notification.configure(text=msg, bg="green", fg="white")
    Notification.place(x=300, y=400)

# ================== GUI Layout ==================
message = tk.Label(window, text="Face Recognition Attendance System", bg="black", fg="white", font=('times', 30, 'bold'))
message.pack(pady=20)

# Enrollment Entry
tk.Label(window, text="Enter Enrollment:", font=('times', 15)).place(x=200, y=200)
txt = tk.Entry(window, font=('times', 15))
txt.place(x=400, y=200)

# Name Entry
tk.Label(window, text="Enter Name:", font=('times', 15)).place(x=200, y=250)
txt2 = tk.Entry(window, font=('times', 15))
txt2.place(x=400, y=250)

# Buttons
tk.Button(window, text="Take Images", command=take_img, font=('times', 15)).place(x=200, y=350)
tk.Button(window, text="Train Model", command=trainimg, font=('times', 15)).place(x=400, y=350)
tk.Button(window, text="Automatic Attendance", command=subjectchoose, font=('times', 15)).place(x=600, y=350)
tk.Button(window, text="Admin Panel", command=admin_panel, font=('times', 15)).place(x=800, y=350)

# Notification area
Notification = tk.Label(window, font=('times', 15))
Notification.place(x=300, y=500)

window.mainloop()

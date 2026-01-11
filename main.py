import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

# Open webcam
video_capture = cv2.VideoCapture(0)

# Load known face images
jaggan_image = face_recognition.load_image_file("Images/jaggan.jpeg")
jaggan_enc = face_recognition.face_encodings(jaggan_image)[0]

mithu_image = face_recognition.load_image_file("Images/Mithu.jpeg")
mithu_enc = face_recognition.face_encodings(mithu_image)[0]

abhay_image = face_recognition.load_image_file("Images/Abhay.jpg")
abhay_enc = face_recognition.face_encodings(abhay_image)[0]

sruthi_image = face_recognition.load_image_file("Images/Sruthi.jpg")
sruthi_enc = face_recognition.face_encodings(sruthi_image)[0]

known_face_encodings = [
    jaggan_enc,
    mithu_enc,
    abhay_enc,
    sruthi_enc
]

known_face_names = [
    "Jaggan",
    "Mithu",
    "Abhay",
    "Sruthi"
]

students = known_face_names.copy()

# Create CSV with today's date
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
f = open(current_date + ".csv", "w+", newline='')
lnwriter = csv.writer(f)

# Main Loop
while True:
    ret, frame = video_capture.read()
    if not ret:
        continue

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert BGR (OpenCV) to RGB (dlib)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)

    if len(face_locations) == 0:
      continue

    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)
        tolerance = 0.5

        name = ""

    if face_distance[best_match_index] < tolerance:
        name = known_face_names[best_match_index]

    if name != "" and name in students:
        students.remove(name)
        current_time = datetime.now().strftime("%H:%M:%S")
        lnwriter.writerow([name, current_time])
        print(name, "marked present at", current_time)


    # Show video
    cv2.imshow("Attendance System", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
f.close()
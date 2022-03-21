'''
Reads the trained face data from the yml file and predicts
the face frames coming from the live webcam.
- If the face is recognized, attendance is recorded for recognized the person name under .csv/.txt files
Implemented by Group3 Team (Monica Dasi, Sahithi R, Nischay Papneja)
Developed for Advanced Real Time Project - WiSe2021/2022
Department of Computer Science, Frankfurt University of Applied Sciences
'''
from tokenize import String
import cv2
import numpy
import os
import pygame
from datetime import datetime
from csv import DictWriter
import pandas as pd
import time

haar_file = 'haarcascade_frontalface_default.xml'
#haar_file = 'haarcascade_frontalface_alt2.xml'
#haar_file = 'lbpcascade_frontalface_improved.xml'

datasets = 'datasets'
font = cv2.FONT_HERSHEY_SIMPLEX
(width, height) = (130, 100)
face_id = 0

recognizer = cv2.face.LBPHFaceRecognizer_create()
# Read the trained faces data from stored yaml file
recognizer.read('./trained_data/trained_faces.yml')

# Use LBPH Recognizer on camera stream
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)

# min window size to be recognized as a face
minW = 0.1*webcam.get(3)
minH = 0.1*webcam.get(4)

img_show = numpy.zeros((height, width, 3), numpy.uint8)
name_rec = 'Face Not Recognized'
is_recognised = False

while True:
    (_, img_cam) = webcam.read()
    gray = cv2.cvtColor(img_cam, cv2.COLOR_BGR2GRAY)
    # gray : Input gray scale image
    # ScaleFactor:
    t1 = time.time()
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=5, minSize=(int(minW), int(minH)),)
    t2 = time.time()

    dt1 = t2 - t1
    print("Time Diff" + str(dt1))

    for (x, y, w, h) in faces:
        padding = 15
        img = cv2.rectangle(img_cam, (x-padding, y-padding),
                            (x+w+padding, y+h+padding), (0, 255, 255), 2)  # BGR

        face = gray[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (width, height))

        # Predict the face data coming from live webcam feed
        # Predict() method takes the captured portion of the face
        # to be analyzed and will return the matched face data.
        face_id, prediction = recognizer.predict(face)
        #print('Face ID : ', face_id)

        df = pd.read_csv('./datasets/employee_data.csv',
                         header=None, index_col=0)
        d = df.to_dict('split')
        d = dict(zip(d['index'], d['data']))
        #print('Stored Dict =>', d)
        name_db = d.get(face_id)[0]  # extract the face name string
        # print(name_db)

        if (prediction < 100):
            img_show = img_cam
            is_recognised = True
            name_rec = name_db
            prediction = "{0}%".format(round(prediction))
            cv2.putText(img_cam, str(prediction), (x+5, y+h-5),
                        font, 1, (255, 255, 0), 1)  # BGR
            (w, h), _ = cv2.getTextSize(
                name_db, font, 0.6, 1)
            img = cv2.rectangle(
                img, (x-15, y - 40), (x + w, y - 20), (0, 255, 0), -1)
            img = cv2.putText(img, name_db, (x-15, y-25),
                              font, 0.6, (0, 0, 0), 1)
        else:
            face_id = "Unknown"
            is_recognised = False
            (w, h), _ = cv2.getTextSize(str(face_id), font, 0.6, 1)
            img = cv2.rectangle(img, (x-15, y - 40),
                                (x + w, y - 20), (51, 51, 255), -1)
            img = cv2.putText(img_cam, str(face_id), (x-15, y-25),
                              font, 0.6, (0, 0, 0), 1)

    cv2.imshow('Face Recognizer', img_cam)

    key = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
    if key == 27:
        print('[INFO] got esc interrupt')
        break

print('[DEBUG] Exited the while loop')
webcam.release()
cv2.destroyAllWindows()

# Lib for playing audio files
pygame.mixer.init()

# path
path = r'D:\RealTimeFaceRecognition\code\datafiles\images\duck_not_recognized.png'

# Reading an image in default mode
image = cv2.imread(path)

print("#######################################################")
print("Real Time Face Recognition System")
print("#######################################################")

# Using cv2.imshow() method display the image and pygame lib to play the music
if is_recognised:
    cv2.imshow(name_rec, img_show)
    pygame.mixer.music.load("./datafiles/audio/success.mp3")
    print('[INFO] Face Recognized Succesfully!')
    print('[INFO] Playing the music now!')
    pygame.mixer.music.play()
else:
    cv2.imshow(name_rec, image)
    pygame.mixer.music.load("./datafiles/audio/failure.mp3")
    print('[INFO] Face NOT Recognized!')
    print('[INFO] Playing the music now!')
    pygame.mixer.music.play()

# waits for user to press any key
# Note: This is necessary to avoid Python kernel form crashing
cv2.waitKey(0)

# Record Attendance to a data file employee_attendance.csv/.txt
if is_recognised:
    count = face_id
    date_time = str(datetime.now())
    fname = name_rec
    emp_no = str(count)
    ar = "[INFO] Recorded the attendance of employee!! \n" + "Emp.ID : " + \
        emp_no + " | Name : " + fname + " | Date and Time : " + date_time

    f = open("./dataFiles/digital_onboarding/employee_attendance.txt", "a")
    f.write(ar)
    f.close()
    print(ar)

    filename = './datafiles/digital_onboarding/employee_attendance.csv'
    file_exists = os.path.isfile(filename)

    # record to a .csv file
    headersCSV = ['S.No', 'Name', 'Date and Time']
    data = {'S.No': emp_no, 'Name': fname, 'Date and Time': date_time}
    with open(filename, 'a', encoding='UTF8', newline='') as f_obj:
        # write the fields
        dict_writer = DictWriter(f_obj, delimiter=',', fieldnames=headersCSV)
        # Pass the data in the dictionary as an argument into the writerow() function
        if not file_exists:
            dict_writer.writeheader()  # If file doesn't exist yet, write the header
        dict_writer.writerow(data)
        # close the file object
        f_obj.close()

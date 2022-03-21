'''
Training Multiple Faces stored on a DataBase:
- Each face should have a unique ID as 1, 2, 3, etc
- LBPH computed model will be saved on trained_faces/ directory. 
Note: If directory does not exist, please create it.
Implemented by Group3 Team (Monica Dasi, Sahithi R, Nischay Papneja)
Developed for Advanced Real Time Project - WiSe2021/2022
Department of Computer Science, Frankfurt University of Applied Sciences
'''

from ctypes import sizeof
import cv2
import numpy as np
import os
import pygame
from time import sleep

# cascade files 'haarcascade' and 'LBP cascade'
haar_file = 'haarcascade_frontalface_default.xml'
#haar_file = 'haarcascade_frontalface_alt2.xml'
#haar_file = 'lbpcascade_frontalface_improved.xml'

recognizer = cv2.face.LBPHFaceRecognizer_create()
classifier = cv2.CascadeClassifier()
classifier.load(haar_file)

# Path for face image database
datasets = 'datasets'

# fetch list of images and a list of corresponding labels
(images, face_ids) = ([], [])
for (subdirs, dirs, files) in os.walk(datasets):

    for subdir in dirs:
        id = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            images.append(cv2.imread(path, 0))
            face_ids.append(int(id))

# Lib for playing audio files
pygame.mixer.init()
pygame.mixer.music.load("./datafiles/audio/training.mp3")
print('[INFO] Playing the music now!')
pygame.mixer.music.play()
sleep(2)

# OpenCV trains a model from the images
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(images, np.array(face_ids))

# Save the model into trained_data/trained_faces.yml
recognizer.write('trained_data/trained_faces.yml')

# Print the number of faces trained
print("#######################################################")
print("Real Time Face Recognition System")
print("#######################################################")
print(
    "\n[INFO] <--- Finished training {0} persons face data --->".format(len(sorted(set(face_ids)))))

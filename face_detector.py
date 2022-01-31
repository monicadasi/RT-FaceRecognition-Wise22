# It helps in identifying the faces
from multiprocessing.connection import wait
import cv2, sys, numpy, os
import pygame
from time import sleep
from datetime import datetime
from csv import DictWriter

size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'
font = cv2.FONT_HERSHEY_SIMPLEX

# Part 1: Create fisherRecognizer
print('Recognizing Face ... please make sure you have sufficient lighting...')

# Create a list of images and a list of corresponding names
(images, labels, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
	for subdir in dirs:
		names[id] = subdir
		subjectpath = os.path.join(datasets, subdir)
		for filename in os.listdir(subjectpath):
			path = subjectpath + '/' + filename
			label = id
			images.append(cv2.imread(path, 0))
			labels.append(int(label))
		id += 1
(width, height) = (130, 100)

# Create a Numpy array from the two lists above
(images, labels) = [numpy.array(lis) for lis in [images, labels]]

# OpenCV trains a model from the images
# NOTE FOR OpenCV2: remove '.face'
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

# Part 2: Use fisherRecognizer on camera stream
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)

img_show = numpy.zeros((height,width,3), numpy.uint8)
name_rec = 'Image OpenCV'
is_recognised = False

while True:
	(_, im) = webcam.read()
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x, y, w, h) in faces:
		cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
		face = gray[y:y + h, x:x + w]
		face_resize = cv2.resize(face, (width, height))

		# Try to recognize the face
		prediction = model.predict(face_resize)
		cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

		if prediction[1] < 100:
			cv2.putText(im,'%s - %.0f' % (names[prediction[0]],prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
			img_show = im
			name_rec = names[prediction[0]]
			is_recognised = True

		# 	# pygame.mixer.music.load("./datafiles/audio/success.mp3")
		# 	# print('Playing the music now!')
		# 	# pygame.mixer.music.play()

		else:
			cv2.putText(im,'Not recognized',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
			is_recognised = False

		# 	# pygame.mixer.music.load("./datafiles/audio/success.mp3")
		# 	# pygame.mixer.music.play()

		
	cv2.imshow('Face Recognizer', im)
	# if is_recognized:
	# 	sleep(10)
	# 	break
		#print(is_recognized)

	# pygame.mixer.music.load("./datafiles/audio/success.mp3")
	# print('Playing the music now!')
	# pygame.mixer.music.play()
	#break
	#cv2.imwrite('opencv'+str(0)+'.png', im)
	
	key = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
	if key == 27:
		print('got esc interrupt')
		break

print('sleep done exited the while loop')
webcam.release()
cv2.destroyAllWindows()

# Lib for playing audio files
pygame.mixer.init()

# path 
path = r'D:\STUDY-FUAS\High_Integrity_Systems\3-Semester-2021-Oct\5_ARTS-Advanced_Real_Time_Systems\RealTimeFaceRecognition\code\datafiles\images\duck_not_recognized.png'
  
# Reading an image in default mode
image = cv2.imread(path)
  
# Window name in which image is displayed
#window_name = 'image'
  
# Using cv2.imshow() method 
# Displaying the image 
if is_recognised:
	cv2.imshow(name_rec, img_show)
	pygame.mixer.music.load("./datafiles/audio/success.mp3")
	print('Playing the music now!')
	pygame.mixer.music.play()
else:
	cv2.imshow(name_rec, image)
	pygame.mixer.music.load("./datafiles/audio/failure.mp3")
	print('Playing the music now!')
	pygame.mixer.music.play()

#waits for user to press any key 
#(this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)

# Module 4 Attendance record in a data file attendance.txt
if is_recognised:
	count = 1
	date_time = str(datetime.now())
	fname = name_rec
	emp_no = str(count)
	ar = "\n" + emp_no + " " + fname + "  " + date_time
	f = open("./dataFiles/attendance/attendance.txt", "a")
	count += 1
	f.write(ar)
	f.close()  
	print(ar)

	filename = './dataFiles/attendance/attendance.csv'
	file_exists = os.path.isfile(filename)

	# record to a .csv file
	headersCSV = ['S.No', 'Name', 'Date and Time']
	data = {'S.No':emp_no, 'Name':fname, 'Date and Time':date_time}
	with open(filename, 'a', encoding='UTF8', newline='') as f_obj:
		#writer = csv.writer(f)
		# write the fields
		dict_writer = DictWriter(f_obj, delimiter=',',fieldnames=headersCSV)
		# Pass the data in the dictionary as an argument into the writerow() function
		
		if not file_exists:
			dict_writer.writeheader()  # file doesn't exist yet, write a header

		dict_writer.writerow(data)
		#close the file object
		f_obj.close()

# print(img_show)
# cv2.imshow('OpenCV', img_show)
# pygame.mixer.music.load("./datafiles/audio/success.mp3")
# print('Playing the music now!')
# pygame.mixer.music.play()

# # Do a bit of cleanup
# print("\n [INFO] Exiting Program and cleanup stuff")
# webcam.release()
# cv2.destroyAllWindows()

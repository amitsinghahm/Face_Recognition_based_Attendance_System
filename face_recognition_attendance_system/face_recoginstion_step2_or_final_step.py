#This is Step2 and Final Step to Recognise faces
# and store the student information in csv file.
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import csv
from datetime import date
import pyttsx3


#Extracting sampled image from the stored folder
data_path = 'E:/face_recognition_training_img/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

#Training these images that are stored in Sampled image folder
Training_Data, Labels = [], []

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Training_Data), np.asarray(Labels))

#print("Model Training Complete!!!!!")

t1 = input('Branch: ')
t2 = input('Subject: ')

face_cascade = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')
def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if faces is ():
        return img, [], 0, 0
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (300,300))

    return img, roi, x, y

#Now Recognising the faces from the webcam
student = []

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    image, face, a, b = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)
        info = onlyfiles[result[0]].split('-')
        roll_no = info[0]
        name = info[1]
        #print('result[0]---> ', result[0])
        #print('onlyfiles---> ', onlyfiles)

        if result[1] < 500:
            confidence = int(100*(1-(result[1])/300))
            display_string = str(confidence)+'%,'+name.capitalize()
        cv2.putText(image, display_string, (a-5, b-5), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 106, 0), 1)

        if confidence>=75:
            cv2.putText(image, 'Unlocked', (250,450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Cropper', image)
            #print(name+' Face found')

            if [roll_no, name] not in student:
                speech = pyttsx3.init()
                speech.say(name+', you are added...')
                speech.runAndWait()
                student.append([roll_no, name])
            print(student)

        else:
            cv2.putText(image, 'Locked', (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', image)
    except:
        cv2.putText(image, 'Face Not Found', (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Cropper', image)

    if cv2.waitKey(1)==13:
        break

cap.release()
cv2.destroyAllWindows()


#Writing student information in "attendance.csv" file
today = [[t1], [t2], [date.today()]]
today.append(['Roll_no','Name'])

with open('attendance.csv', 'a') as att:
    write = csv.writer(att)
    write.writerows(today)
    write.writerows(student)

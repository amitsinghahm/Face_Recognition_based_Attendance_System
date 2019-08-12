#This is 1st step of Face Recognition based Attendance System
#This step is taking a Face Sampling from webcam.
#Sampled faces are stored in "file_path_name" of variable in a folder.

import cv2

roll_no = input("What is your rollno??\n")
name = input("What is your name??\n")

face_cascade = cv2.CascadeClassifier("../haarcascade_frontalface_default.xml")


def face_extracter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if faces is ():
        return None
    for (x,y,w,h) in faces:
        cropped_faces = img[y:y+h, x:x+w]
    return(cropped_faces)


cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if face_extracter(frame) is not None:
        count+= 1
        face = cv2.resize(face_extracter(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        file_path_name = "E:/face_recognition_training_img/"+roll_no+"-"+name+"-"+str(count)+".jpg"
        cv2.imwrite(file_path_name, face)

        cv2.putText(face,str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Cropper', face)

    else:
        print('Face Not Found')

    if cv2.waitKey(1) == 13 or count == 50:
        break

cap.release()
cv2.destroyAllWindows()
print('Collecting Sample Complete!!!')


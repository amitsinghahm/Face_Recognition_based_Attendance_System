#importing OpenCV
import cv2

#Create Cascade classifier Object
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#Reading image in RGB format
#img = cv2.imread("img/DSC_1468.JPG", 1)
img = cv2.imread("img/DSC_1452.JPG", 1)
#img = cv2.imread("img/DSC_1396.JPG", 1)
#img = cv2.imread("img/mahendra-singh-dhoni1.jpg", 1)

#Convert RGB image into Gray scale image
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Search the image co-ordinate
faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.27978, minNeighbors = 5)

for x,y,w,h in faces:
    img = cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 3)

#Resizing the image
resized = cv2.resize(img, (int(img.shape[1]/2),int(img.shape[0]/2)))

#Showing image
cv2.imshow("Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
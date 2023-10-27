import cv2 as cv

face = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv.imread('c.JPG')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
f = face.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in f:
    cv.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 3)

cv.imshow('img', img)
cv.waitKey()
import cv2
import numpy as np
import face_recognition

imgChandan = face_recognition.load_image_file('Images/chandan.jpg')
imgChandan = cv2.cvtColor(imgChandan, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('Images/Test2.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLocation = face_recognition.face_locations(imgChandan)[0]
encodeChandan = face_recognition.face_encodings(imgChandan)[0]
cv2.rectangle(imgChandan, (faceLocation[3], faceLocation[0]), (faceLocation[1], faceLocation[2]), (255, 0, 255), 3)

faceLocationTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocationTest[3], faceLocationTest[0]), (faceLocationTest[1], faceLocationTest[2]), (255, 0, 255), 3)

results = face_recognition.compare_faces([encodeChandan], encodeTest)
faceDistance = face_recognition.face_distance([encodeChandan], encodeTest)
print(results, faceDistance)
cv2.putText(imgTest, f'{results} {round(faceDistance[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Chandan Kumar', imgChandan)
cv2.imshow('Chandan Test', imgTest)
cv2.waitKey(0)

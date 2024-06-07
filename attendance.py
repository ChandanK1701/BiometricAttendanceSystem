import time
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, date
import pygame

# Initialize pygame mixer for sound
pygame.mixer.init()
duplicate_sound = pygame.mixer.Sound("Music/duplicate punch.mp3")  # Replace "duplicate_punch_sound.wav" with your sound file
sound_played = False


present_names = []
present_dates = []
path = 'AttendanceImages'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cls in myList :
    currentImage = cv2.imread(f'{path}/{cls} ')
    images.append(currentImage)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeImage = face_recognition.face_encodings(img)[0]
        encodeList.append(encodeImage)
    return encodeList



def markAttendance(name):
    global sound_played
    if name not in present_names and date.today() not in present_dates:
        with open('Attendance.csv', 'r+') as f:
            myDataList = f.readline()
            nameList = []
            for line in myDataList:
                entry = line.split(',')
                nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dateString = now.date()
                timeString = now.strftime('%H:%M:%S')
                f.writelines(f'\n {name}, {dateString}, {timeString}')
                present_names.append(name)
                present_dates.append(date.today())
                return True
            else:
                print(f"{name} is already marked present.")
                return False
    else:
        print(f"Duplicate punch for {name}.")
        if not sound_played:
            time.sleep(10)
            duplicate_sound.play()  # Play sound for duplicate punch
            sound_played = True
        return False

        # print(myDataList)

encodeListKnownFaces = findEncodings(images)
print('Encoding Complete')

capture = cv2.VideoCapture(0)
while True:
    success, img = capture.read()
    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    facesInCurrentFrame = face_recognition.face_locations(imgSmall)
    encodesCurrentFrame = face_recognition.face_encodings(imgSmall, facesInCurrentFrame)

    for encodeFace, faceLocation in zip(encodesCurrentFrame, facesInCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnownFaces, encodeFace)
        faceDistance = face_recognition.face_distance(encodeListKnownFaces, encodeFace)
        print(faceDistance)
        matchIndex = np.argmin(faceDistance)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            #cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            if markAttendance(name):
                cv2.putText(img, "Marked Present", (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            else:
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#cap.release()
cv2.destroyAllWindows()

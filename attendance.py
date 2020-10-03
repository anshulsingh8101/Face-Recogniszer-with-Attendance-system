import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

#1
path = r'C:\Users\Anshul Singh\PycharmProjects\finalproject\imageAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

#2
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')           #current Image
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

#3
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
encodeListKnown = findEncodings(images)
print('Encoding Complete')


#4
def markAttendance(name):
    with open('attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')   # string format for time
            f.writelines(f'\n{name},{dtString}')

#4 video
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)           #done to prevent from multiple faces error
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)             #on the current image frame and the location it gives the encoding

    # Step for finding our matches
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)            # returns a list of distance b/w the faces and the video faces
        print(faceDis)
        matchIndex = np.argmin(faceDis)      # We take the minimum value

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    k = cv2.waitKey(1)
    if k == 27:                     #esc button
        break

cap.release()
cv2.destroyAllWindows()

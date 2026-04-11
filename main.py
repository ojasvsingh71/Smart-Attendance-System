import cv2
import numpy as np
import face_recognition
import os

#Load image
path='images'
images=[]
classNames=[]

myList=os.listdir(path)

for cl in myList:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

#Encode faces
def findEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown=findEncodings(images)
print("Encoding Complete")

#Start webcam
cap=cv2.VideoCapture(0)
from datetime import datetime

import csv
from datetime import datetime

def markAttendance(name):
    now=datetime.now()
    date=now.strftime('%Y-%m-%d')
    time=now.strftime('%H:%M:%S')

    with open('attendance.csv','r+') as f:
        reader=csv.reader(f)
        data=list(reader)

        already_marked=False

        for row in data:
            if len(row)>0 and row[0] == name and row[1]==date:
                already_marked = True
                break

        if not already_marked:
            writer = csv.writer(f)
            writer.writerow([name, date, time])
            
while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img, name, (x1,y2+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            markAttendance(name)
    cv2.imshow("Recognition", img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
import face_recognition
import os
import json
import csv
from datetime import datetime
from urllib import error, request

#Load image
path='images'
images=[]
classNames=[]
backendBaseUrl = os.getenv('BACKEND_URL', 'http://127.0.0.1:8000')
markedToday = set()
backendUnavailableLogged = False
backendAvailableAtStartup = False

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


def markAttendanceLocal(name):
    now = datetime.now()
    date = now.strftime('%Y-%m-%d')
    time = now.strftime('%H:%M:%S')

    alreadyMarked = False
    with open('attendance.csv', 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) > 1 and row[0] == name and row[1] == date:
                alreadyMarked = True
                break

    if not alreadyMarked:
        with open('attendance.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, date, time])


def markAttendanceBackend(name):
    payload = json.dumps({"name": name}).encode('utf-8')
    req = request.Request(
        f'{backendBaseUrl}/attendance/mark',
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )

    with request.urlopen(req, timeout=2.0) as response:
        body = response.read().decode('utf-8')
        return json.loads(body)


def checkBackendHealth():
    req = request.Request(f'{backendBaseUrl}/health', method='GET')
    with request.urlopen(req, timeout=2.0) as response:
        body = response.read().decode('utf-8')
        data = json.loads(body)
        return data.get('status') == 'ok'

def markAttendance(name):
    global backendUnavailableLogged

    date = datetime.now().strftime('%Y-%m-%d')
    cacheKey = f'{name}:{date}'
    if cacheKey in markedToday:
        return

    try:
        result = markAttendanceBackend(name)
        if result.get('message') in ('Attendance marked', 'Already marked'):
            markedToday.add(cacheKey)
    except (error.URLError, TimeoutError, json.JSONDecodeError):
        # Fallback keeps recognition usable even when the API is down.
        if not backendUnavailableLogged:
            print('Backend unavailable, writing attendance to local CSV fallback.')
            backendUnavailableLogged = True
        markAttendanceLocal(name)
        markedToday.add(cacheKey)


try:
    backendAvailableAtStartup = checkBackendHealth()
except (error.URLError, TimeoutError, json.JSONDecodeError):
    backendAvailableAtStartup = False

if backendAvailableAtStartup:
    print(f'Backend connected at {backendBaseUrl}. Attendance will be marked via API.')
else:
    print(f'Backend offline at {backendBaseUrl}. Attendance will use local CSV fallback.')
            
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
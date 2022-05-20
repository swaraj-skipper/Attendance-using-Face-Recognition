import face_recognition
import cv2 
import numpy as np
import os
from datetime import datetime

images = []
names = []
path = 'images'

mlistOfNames = os.listdir(path)

for i in mlistOfNames:
    current_img = cv2.imread(f'{path}/{i}')
    images.append(current_img)
    names.append(os.path.splitext(i)[0])

def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
#algorithm = hog transformation used for encodings

encodedList = faceEncodings(images)
print("All Encodings Complete!!")

def attendance(name):
    with open('Attendance.xsl','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            firstName = line.split(',')
            nameList.append(firstName[0])

        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'{name},{tStr},{dStr}\n')
            


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    faces = cv2.resize(frame, (0,0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces,facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodedList,encodeFace)
        faceDis = face_recognition.face_distance(encodedList,encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = names[matchIndex].upper()
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(frame,name,(x1+6,y2-6),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
            attendance(name)

    cv2.imshow("Camera",frame)
    if cv2.waitKey(10) == 27:
        break


cap.release()
cv2.destroyAllWindows()
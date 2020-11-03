import pickle
import os
import cv2
import numpy as np
from keras.engine.saving import load_model

face_cas=cv2.CascadeClassifier("C:\\Users\\Adarsh\\anaconda3\\envs\\facerecog\\Library\\etc\\haarcascades\\haarcascade_frontalface_alt2.xml")

#face_cas=cv2.CascadeClassifier("C:\\Users\\Adarsh\\anaconda3\\envs\\facerecog\\Library\\etc\\haarcascades\\haarcascade_profileface.xml")
#recognizer=cv2.face.LBPHFaceRecognizer_create()
#recognizer.read("trainner.yml")
model=load_model('first.h5')
with open("labels.pickle","rb") as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}

cap=cv2.VideoCapture(0)

cap.set(3,840)
cap.set(4,620)
#cap.set(10,100)

while True:
    success,img=cap.read()
    img=cv2.flip(img,1)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cas.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    #faces1=face_cas1.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for x, y, w, h in faces:
        # print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y + h, x:x + w]
        roi_gray=cv2.resize(roi_gray,(500,500))
        roi_gray=np.expand_dims(roi_gray,axis=0)
        roi_gray = np.expand_dims(roi_gray, -1)
        confidence = model.predict(roi_gray)
        print(confidence[0][0])
        if (confidence[0][0] > 0.5):
            name = 'Adarsh'
            cv2.putText(img, name, (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        elif(confidence[0][1]>0.5):
            name='Unknown'
            cv2.putText(img, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            #print(id_)
            #print(labels[id_])

        img_item="my_image.png"
        cv2.imwrite(img_item,roi_gray)
        cord_x=x+w
        cord_y=y+h
        color=(255,0,0)
        stroke=4
        cv2.rectangle(img,(x,y),(cord_x,cord_y),color,stroke)
    cv2.imshow("Video",img)
    if cv2.waitKey(1) % 0XFF==ord('q'):
        breakq
import os
import pickle
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense, Convolution2D
from keras.optimizers import Adam
from keras.utils import to_categorical

WIDTH = 300
HEIGHT = 300
images = []
face_cas = cv2.CascadeClassifier(
    'C:\\Users\\Adarsh\\anaconda3\\envs\\facerecog\\Library\\etc\\haarcascades\\haarcascade_frontalface_alt2.xml')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(BASE_DIR, "Images")
recognizer = cv2.face.LBPHFaceRecognizer_create()
x_train = []
y_labels = []
current_id = 0
label_ids = dict()
for root, dirs, files in os.walk(img_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "_").lower()
            print(label, path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            pil_image = cv2.imread(path)
            gray = cv2.cvtColor(pil_image, cv2.COLOR_BGR2GRAY)
            image_array = np.asarray(gray, dtype=np.uint8)
            #image_array=image_array.reshape((342,342,1))
            x_train.append(image_array)
            y_labels.append(id_)
            faces = face_cas.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            '''for x, y, w, h in faces:
                roi = image_array[y:y + h, x:x + w]
                print(roi.shape)
                # roi=roi.reshape((-1,300,300,1))
                x_train.append(roi)
                # plt.imshow(roi)
                # plt.show()
                y_labels.append(id_)'''
print(y_labels)
y = to_categorical(y_labels)
x = tf.convert_to_tensor(x_train)
x = np.expand_dims(x, -1)
with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)
# print(x.shape[0:])
model = Sequential()

model.add(Conv2D(16, 3, 3, border_mode='same', batch_size=21, input_shape=(500, 500, 1)))
model.add(Activation('relu'))
model.add(Conv2D(16, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Conv2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()
model.build(input_shape=(500, 500, 1))
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
model.fit(x, y, steps_per_epoch=2, epochs=30)

model.save("first.h5")
# recognizer.train(x_train,np.array(y_labels))
# recognizer.save("trainner.yml")

import os
import cv2
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout

dataset_path = r"leapGestRecog\leapGestRecog"

images = []
labels = []
gesture_labels = {}

label = 0

print("Loading dataset...")

for person in os.listdir(dataset_path):

    person_path = os.path.join(dataset_path, person)

    if not os.path.isdir(person_path):
        continue

    for gesture in os.listdir(person_path):

        gesture_path = os.path.join(person_path, gesture)

        if not os.path.isdir(gesture_path):
            continue

        if gesture not in gesture_labels:
            gesture_labels[gesture] = label
            label += 1

        current_label = gesture_labels[gesture]

        for file in os.listdir(gesture_path):

            path = os.path.join(gesture_path, file)

            if not file.lower().endswith(
                (".jpg",".jpeg",".png")
            ):
                continue

            img = cv2.imread(path)

            if img is None:
                continue

            img = cv2.resize(img,(64,64))
            img = img.astype("float32")/255.0

            images.append(img)
            labels.append(current_label)

print("Dataset Loaded")
print("Images:",len(images))

X=np.array(images)

y=to_categorical(np.array(labels))

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training started...")

model=Sequential()

model.add(
    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(64,64,3)
    )
)

model.add(MaxPooling2D((2,2)))

model.add(
    Conv2D(
        64,
        (3,3),
        activation='relu'
    )
)

model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128,activation='relu'))

model.add(Dropout(0.5))

model.add(
    Dense(
        y.shape[1],
        activation='softmax'
    )
)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32
)

loss,accuracy=model.evaluate(
    X_test,
    y_test
)

print("Accuracy:",accuracy)

model.save("hand_gesture_model.h5")

with open("gesture_labels.pkl","wb") as f:
    pickle.dump(gesture_labels,f)

print("Model and labels saved")
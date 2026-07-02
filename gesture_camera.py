import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("hand_gesture_model.h5")

gesture_names = [
    'Palm','L','Fist','Fist_moved',
    'Thumb','Index','OK',
    'Palm_moved','C','Down'
]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# High quality camera settings
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
cap.set(cv2.CAP_PROP_FPS,30)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame,1)

    # Mild brightness only
    frame = cv2.convertScaleAbs(
        frame,
        alpha=1.1,
        beta=20
    )

    x1,y1=150,100
    x2,y2=450,400

    cv2.rectangle(
        frame,
        (x1,y1),
        (x2,y2),
        (0,255,0),
        2
    )

    roi = frame[y1:y2,x1:x2]

    img = cv2.resize(roi,(64,64))
    img = img.astype("float32")/255.0
    img = np.expand_dims(img,axis=0)

    prediction = model.predict(img,verbose=0)

    result = np.argmax(prediction)
    confidence = np.max(prediction)*100

    text = f"{gesture_names[result]} ({confidence:.1f}%)"

    cv2.putText(
        frame,
        text,
        (50,60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow(
        "Hand Gesture Recognition",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
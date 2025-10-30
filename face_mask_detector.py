import cv2
import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Flatten, Dense # type: ignore

model = Sequential([
    Flatten(input_shape=(64, 64, 3)),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    face = cv2.resize(frame, (64,64))
    face = face.astype('float') / 255.0
    face = np.expand_dims(face, axis=0)

    pred = model.predict(face)[0]
    label = "Mask" if pred[0] > pred[1] else "No Mask"
    color = (0,255,0) if label == "Mask" else (0,0,255)

    cv2.putText(frame, label, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Mask Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
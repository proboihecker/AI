import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2)
cap = cv2.VideoCapture(0)

def fingers_up(handLms):
    tips = [4, 8 , 12, 16, 20]
    fingers = []
    for i, tip in enumerate(tips):
        if i == 0:
            if handLms.landmark[tip].x < handLms.landmark[tip-1].x:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if handLms.landmark[tip].y < handLms.landmark[tip-2].y:
                fingers.append(1)
            else:
                fingers.append(0)

    return fingers

gestures = {
    (0,1,0,0,0): "Pointing",
    (0,1,1,0,0): "Peace",
    (1,1,1,1,1): "Open Hand",
    (0,0,0,0,0): "Fist",
}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)


    gesture = ""
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            status = tuple(fingers_up(handLms))
            gesture = gestures.get(status, "")

    cv2.putText(frame, gesture, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
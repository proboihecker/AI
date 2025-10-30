import cv2
import mediapipe as mp
from pynput.mouse import Controller, Button
import platform
import subprocess

# --- MediaPipe setup ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# --- Mouse controller ---
mouse = Controller()

# --- Screen size ---
if platform.system() == "Windows":
    import ctypes
    screen_w = ctypes.windll.user32.GetSystemMetrics(0)
    screen_h = ctypes.windll.user32.GetSystemMetrics(1)
else:
    import pyautogui
    screen_w, screen_h = pyautogui.size()

# --- Video capture ---
cap = cv2.VideoCapture(0)

prev_x, prev_y = 0, 0
smooth = 5
prev_scroll_y = None

# --- Helper functions ---
def dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

def ydotool_move(x, y):
    subprocess.run(['sudo', 'ydotool', 'mousemove', str(int(x)), str(int(y))])

def ydotool_click(button='left'):
    btn = '1' if button == 'left' else '2'
    subprocess.run(['sudo', 'ydotool', 'click', btn])

def ydotool_scroll(amount):
    subprocess.run(['sudo', 'ydotool', 'scroll', '0', str(amount)])

# --- Main loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            landmarks = handLms.landmark
            thumb = landmarks[4]
            index_fin = landmarks[8]
            middle_fin = landmarks[12]

            x = int(index_fin.x * screen_w)
            y = int(index_fin.y * screen_h)

            curr_x = prev_x + (x - prev_x) / smooth
            curr_y = prev_y + (y - prev_y) / smooth
            prev_x, prev_y = curr_x, curr_y

            mouse.position = (curr_x, curr_y)

            # Uncomment below for gestures
            if dist(thumb, index_fin) < 0.05:
                mouse.click(Button.left, 1)
                cv2.putText(frame, 'Left Click', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            # elif dist(thumb, middle_fin) < 0.05:
            #     mouse.click(Button.right, 1)
            #     cv2.putText(frame, 'Right Click', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            # elif dist(index_fin, middle_fin) < 0.05:
            #     curr_scroll_y = index_fin.y + middle_fin.y / 2
            #     if prev_scroll_y is not None:
            #         dy = prev_scroll_y - curr_scroll_y
            #         scroll_amount = int(dy * 50)
            #         if scroll_amount != 0:
            #             mouse.scroll(0, scroll_amount)
            #             cv2.putText(frame, 'Scroll', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            #     prev_scroll_y = curr_scroll_y

    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

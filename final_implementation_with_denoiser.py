import cv2
import numpy as np
import time
import screeninfo
from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KeyboardController, Key

import htm2 as htm  # Your custom hand tracking module
import mediapipe as mp

# Denoiser imports
import torch
from torchvision import transforms
from denoiser_import import UNet

# Setup camera
cap = cv2.VideoCapture(0)

# Constants
CROP_WIDTH, CROP_HEIGHT = 1000, 1000
CROP_PADDING = 200
wcam, hcam = 1280, 720
frameR = 100
smoothening = 10
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# Control setup
mouse = MouseController()
keyboard = KeyboardController()
screen = screeninfo.get_monitors()[0]
screen_width, screen_height = screen.width, screen.height

# Hand detector
detector = htm.handDetector(detectionCon=0.3, maxHands=1,trackCon=0.3)

# Mediapipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
last_crop_time = 0
crop_interval = 1.5
crop_box = None

# Denoiser setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
denoiser = UNet().to(device)
denoiser.load_state_dict(torch.load("/Users/sadik2/Desktop/cv_project/checkpoints/unet_epoch3.pth", map_location=device))
denoiser.eval()
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()
last_denoise_time = 0
denoise_interval = 60

# Keyboard UI
keys = [
    list("QWERTYUIOP"),
    list("ASDFGHJKL"),
    list("ZXCVBNM"),
    ["SPACE", "BACK", "LEFT", "RIGHT", "UP", "DOWN"]
]
key_size = 80
keyboard_origin = (wcam // 2 - 6 * (key_size + 5), 100)
selected_key = None
selected_time = 0
mode = 'Mouse'
last_toggle_time = 0

def draw_keyboard(img):
    key_pos = {}
    overlay = img.copy()
    for i, row in enumerate(keys):
        for j, key in enumerate(row):
            x = keyboard_origin[0] + j * (key_size + 5)
            y = keyboard_origin[1] + i * (key_size + 5)
            cv2.rectangle(overlay, (x, y), (x + key_size, y + key_size), (200, 200, 200), cv2.FILLED)
            cv2.rectangle(img, (x, y), (x + key_size, y + key_size), (0, 0, 0), 2)
            font_scale = 1 if len(key) == 1 else 0.6
            text_x = x + 10 if len(key) == 1 else x + 5
            text_y = y + 50
            cv2.putText(img, key, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, font_scale, (0, 0, 0), 2)
            key_pos[key] = (x, y, x + key_size, y + key_size)
    alpha = 0.3
    cv2.addWeighted(img, 0.5, img, 0.5, 0, img)
    return key_pos

def detect_key_press(lmList, key_pos):
    global selected_key, selected_time
    if not lmList:
        return None
    x, y = lmList[8][1:]
    for key, (x1, y1, x2, y2) in key_pos.items():
        if x1 < x < x2 and y1 < y < y2:
            if selected_key == key:
                if time.time() - selected_time > 1:
                    selected_key = None
                    return key
            else:
                selected_key = key
                selected_time = time.time()
    return None

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    current_time = time.time()

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        x_vals = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x]
        y_vals = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                  landmarks[mp_pose.PoseLandmark.LEFT_HIP].y,
                  landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
        if current_time - last_crop_time > crop_interval:
            x_min, x_max = int(min(x_vals) * frame.shape[1]), int(max(x_vals) * frame.shape[1])
            y_min, y_max = int(min(y_vals) * frame.shape[0]), int(max(y_vals) * frame.shape[0])
            cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2

            x1 = max(0, cx - (CROP_WIDTH // 2 + CROP_PADDING))
            y1 = max(0, cy - (CROP_HEIGHT // 2 + CROP_PADDING))
            x2 = min(frame.shape[1], cx + (CROP_WIDTH // 2 + CROP_PADDING))
            y2 = min(frame.shape[0], cy + (CROP_HEIGHT // 2 + CROP_PADDING))

            crop_box = (x1, y1, x2, y2)
            last_crop_time = current_time

    if crop_box:
        x1, y1, x2, y2 = crop_box
        cropped_frame = frame[y1:y2, x1:x2]
        cropped_frame = cv2.resize(cropped_frame, (wcam, hcam))

        # ⏱ Apply denoising only once every 6 seconds
        if current_time - last_denoise_time > denoise_interval:
            with torch.no_grad():
                input_tensor = to_tensor(cropped_frame).unsqueeze(0).to(device)
                denoised_tensor = denoiser(input_tensor).squeeze(0).cpu()
                cropped_frame = np.array(to_pil(denoised_tensor)).astype(np.uint8)
            last_denoise_time = current_time
    else:
        cropped_frame = cv2.resize(frame, (wcam, hcam))

    img = detector.findHands(cropped_frame)
    lmList, bbox = detector.findPosition(img)
    fingers = detector.fingersUp() if lmList else []

    if fingers and fingers[0] == 1 and fingers[4] == 1:
        if time.time() - last_toggle_time > 1.5:
            mode = 'Keyboard' if mode == 'Mouse' else 'Mouse'
            last_toggle_time = time.time()

    if mode == 'Mouse':
        cv2.putText(img, 'Mode: Mouse', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        if lmList:
            x1, y1 = lmList[8][1:]
            if fingers[1] == 1 and fingers[2] == 0:
                x3 = np.interp(x1, (frameR, wcam - frameR), (0, screen_width))
                y3 = np.interp(y1, (frameR, hcam - frameR), (0, screen_height))
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening
                mouse.position = (screen_width - clocX, clocY)
                plocX, plocY = clocX, clocY

            if fingers[1] == 1 and fingers[2] == 1:
                length, img, _ = detector.findDistance(8, 12, img)
                if length < 40:
                    mouse.click(Button.left)
                    time.sleep(0.2)

    elif mode == 'Keyboard':
        cv2.putText(img, 'Mode: Keyboard', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        key_pos = draw_keyboard(img)
        key = detect_key_press(lmList, key_pos)
        if key:
            if len(key) == 1:
                keyboard.press(key.lower())
                keyboard.release(key.lower())
            elif key == 'SPACE':
                keyboard.press(Key.space)
                keyboard.release(Key.space)
            elif key == 'BACK':
                keyboard.press(Key.backspace)
                keyboard.release(Key.backspace)
            elif key == 'LEFT':
                keyboard.press(Key.left)
                keyboard.release(Key.left)
            elif key == 'RIGHT':
                keyboard.press(Key.right)
                keyboard.release(Key.right)
            elif key == 'UP':
                keyboard.press(Key.up)
                keyboard.release(Key.up)
            elif key == 'DOWN':
                keyboard.press(Key.down)
                keyboard.release(Key.down)

    cv2.imshow("Virtual Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

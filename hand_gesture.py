import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Detect both hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)
    total_fingers = 0

    if results.multi_hand_landmarks:
        for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):

            landmarks = []
            h, w, c = img.shape

            # Collect landmark positions
            for lm in hand_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append((cx, cy))

            fingers = 0

            # Detect Left or Right Hand
            hand_label = results.multi_handedness[hand_no].classification[0].label

            # 👍 Thumb (different logic for left & right hand)
            if hand_label == "Right":
                if landmarks[4][0] > landmarks[3][0]:
                    fingers += 1
            else:  # Left hand
                if landmarks[4][0] < landmarks[3][0]:
                    fingers += 1

            # ✋ Other 4 fingers (same logic)
            finger_tips = [8, 12, 16, 20]
            for tip in finger_tips:
                if landmarks[tip][1] < landmarks[tip-2][1]:
                    fingers += 1

            total_fingers += fingers

            # Draw hand landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show finger count
    cv2.putText(img, f'Total Fingers: {total_fingers}', (30,100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

    cv2.imshow("Hand Gesture - 10 Fingers", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import HandDetectionMin as hdm
import mediapipe as mp
from gpiozero import LED
import time

led1 = LED(2)
led2 = LED(3)
led3 = LED(4)

# wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
# cap.set(3, wCam)
# cap.set(4, hCam)
pTime = 0

detector = hdm.HandDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # Check the state of fingers
        index_finger_open = lmList[tipIds[1]][2] < lmList[tipIds[1] - 2][2]
        middle_finger_open = lmList[tipIds[2]][2] < lmList[tipIds[2] - 2][2]
        ring_finger_open = lmList[tipIds[3]][2] < lmList[tipIds[3] - 2][2]

        # Control LEDs based on finger states
        if index_finger_open and not middle_finger_open and not ring_finger_open:
            led1.on()
            led2.off()
            led3.off()
        elif index_finger_open and middle_finger_open and not ring_finger_open:
            led1.off()
            led2.on()
            led3.off()
        elif index_finger_open and middle_finger_open and ring_finger_open:
            led1.off()
            led2.off()
            led3.on()
        else:
            led1.off()
            led2.off()
            led3.off()

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS : {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

import cv2
import mediapipe as mp 
import time

class HandDetector():
    def __init__(self, mode=False, maxHands= 4, detectionCon= 0.5, trackCon= 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands, 
                                        min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon
    )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)
        # print(result.multi_hand_landmarks)

        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                    
        return img
    
    def findPosition(self, img, handNo=0, draw=True):

        lmList = []

        if self.result.multi_hand_landmarks:
            myHands = self.result.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHands.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
                    

        return lmList
                
def main():
    pTime = 0
    cTime = 0

    detector = HandDetector()
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()

        # Check if frame capture is successful
        if not success:
            print("Failed to capture frame. Exiting...")
            break

        # Check if the image is empty
        if img is None:
            print("Empty image. Exiting...")
            break

        img = detector.findHands(img)

        lmlist = detector.findPosition(img)
        # if len(lmlist) != 0:
        #     print([lmlist[4]])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
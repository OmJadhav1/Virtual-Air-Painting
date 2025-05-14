import cv2
import mediapipe as mp

class handDetector():
    def __init__(self, static_image_mode=False, max_num_hands=1, 
                 min_detection_confidence=0.7, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(
            static_image_mode=self.static_image_mode,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.mpdraw = mp.solutions.drawing_utils
        self.finger_tip_id = [4, 8, 12, 16, 20]
        self.lm_list = []

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        self.results = results  # Store results for later use
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(img, hand_landmarks, 
                                             self.mphands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, hand_num=0, draw=True):
        self.lm_list = []
        if self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) > hand_num:
                myHand = self.results.multi_hand_landmarks[hand_num]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lm_list.append([id, cx, cy])
                    if draw and id in self.finger_tip_id:
                        cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
        return self.lm_list

    def fingerStatus(self):
        fingers = [0] * 5
        if len(self.lm_list) >= 21:
            # Thumb detection based on handedness
            if self.results.multi_handedness:
                handedness = self.results.multi_handedness[0].classification[0].label
                if handedness == 'Left':
                    fingers[0] = 1 if self.lm_list[4][1] > self.lm_list[3][1] else 0
                else:
                    fingers[0] = 1 if self.lm_list[4][1] < self.lm_list[3][1] else 0
            
            # Other fingers
            for i in range(1, 5):
                fingers[i] = 1 if self.lm_list[self.finger_tip_id[i]][2] < \
                                self.lm_list[self.finger_tip_id[i]-2][2] else 0
        return fingers
import cv2
import numpy as np
import os
import track_hands as TH

class VideoCamera():
    def __init__(self, overlay_image=[], draw_color=(255, 200, 100)):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)
        self.xp = 0
        self.yp = 0
        self.brush_thickness = 15
        self.eraser_thickness = 100
        self.overlay_image = overlay_image
        self.draw_color = draw_color
        self.detector = TH.handDetector(
            min_detection_confidence=0.7,
            static_image_mode=False  # Better for video
        )
        self.image_canvas = np.zeros((720, 1280, 3), np.uint8)
        self.default_overlay = overlay_image[0]
        self.color_regions = [
            (0, 200, 0, overlay_image[0], (255, 0, 0)),
            (200, 400, 1, overlay_image[1], (47, 225, 245)),
            (400, 600, 2, overlay_image[2], (197, 47, 245)),
            (600, 800, 3, overlay_image[3], (53, 245, 47)),
            (1100, 1280, 4, overlay_image[4], (0, 0, 0))
        ]

    def __del__(self):
        self.cap.release()

    def get_frame(self):
        success, frame = self.cap.read()
        if not success:
            return b''
            
        frame = cv2.flip(frame, 1)
        frame = self.detector.findHands(frame)
        landmark_list = self.detector.findPosition(frame, draw=False)

        # Process hand landmarks
        if landmark_list:
            x1, y1 = landmark_list[8][1:]  # Index finger
            x2, y2 = landmark_list[12][1:]  # Middle finger
            fingers = self.detector.fingerStatus()

            # Color Selection Mode
            if fingers[1] and fingers[2]:
                self.xp, self.yp = 0, 0
                if y1 < 125:
                    for x_start, x_end, idx, overlay, color in self.color_regions:
                        if x_start <= x1 < x_end:
                            self.default_overlay = overlay
                            self.draw_color = color
                            break
                cv2.putText(frame, 'Color Selector', (900, 680), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.line(frame, (x1, y1), (x2, y2), self.draw_color, 3)

            # Drawing Mode
            elif fingers[1] and not fingers[2]:
                if self.xp == 0 and self.yp == 0:
                    self.xp, self.yp = x1, y1

                thickness = self.eraser_thickness if self.draw_color == (0, 0, 0) else self.brush_thickness
                cv2.line(self.image_canvas, (self.xp, self.yp), (x1, y1), 
                        self.draw_color, thickness)

                self.xp, self.yp = x1, y1
                cv2.putText(frame, 'Drawing', (900, 680), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Merge canvas and frame
        frame[0:125, 0:1280] = self.default_overlay
        img_gray = cv2.cvtColor(self.image_canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, img_inv)
        frame = cv2.bitwise_or(frame, self.image_canvas)

        # Optimized encoding
        ret, jpeg = cv2.imencode('.jpg', frame, 
                               [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        return jpeg.tobytes()

def main():
    overlay_image = []
    header_img = "Images"
    header_img_list = os.listdir(header_img)
    for img in header_img_list:
        overlay_image.append(cv2.imread(f'{header_img}/{img}'))

    cam1 = VideoCamera(overlay_image=overlay_image)

    while True:
        frame_bytes = cam1.get_frame()
        if cv2.waitKey(1) == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
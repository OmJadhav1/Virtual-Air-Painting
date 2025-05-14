import cv2
import numpy as np
import time
import os
import track_hands as TH

# Configuration
BRUSH_THICKNESS = 15
ERASER_THICKNESS = 100
COLORS = [
    (255, 0, 0),    # Blue
    (47, 225, 245), # Cyan
    (197, 47, 245), # Pink
    (53, 245, 47),  # Green``
    (0, 0, 0)       # Eraser (Black)
]
COLOR_NAMES = ["Blue", "Cyan", "Pink", "Green", "Eraser"]
MENU_HEIGHT = 100  # Height of the menu bar 

# Kalman Filter Class
class KalmanFilter:
    def __init__(self, process_noise=1e-3, measurement_noise=1e-1):
        self.kf = cv2.KalmanFilter(4, 2)  # 4 state variables, 2 measurement variables
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise

    def update(self, x, y):
        """Update the Kalman Filter with a new measurement."""
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measurement)
        prediction = self.kf.predict()
        return int(prediction[0]), int(prediction[1])

def draw_menu(frame):
    """Draw the dynamic menu at the top of the screen."""
    menu_width = frame.shape[1] // len(COLORS)
    for i, color in enumerate(COLORS):
        x_start = i * menu_width
        x_end = (i + 1) * menu_width
        cv2.rectangle(frame, (x_start, 0), (x_end, MENU_HEIGHT), color, -1)
        cv2.putText(frame, COLOR_NAMES[i], (x_start + 10, MENU_HEIGHT - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame

def main():
    # Initialization
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    # Create resizable window
    cv2.namedWindow('Paint', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Paint', 1280, 720)  # Initial size

    detector = TH.handDetector(min_detection_confidence=0.7)
    image_canvas = np.zeros((720, 1280, 3), np.uint8)
    
    xp, yp = 0, 0
    draw_color = COLORS[0]  # Start with the first color
    previousT = time.time()
    clear_canvas_flag = False  # To track if canvas was cleared

    # Initialize Kalman Filters
    kf_index = KalmanFilter()
    kf_middle = KalmanFilter()

    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        frame = detector.findHands(frame)
        landmark_list = detector.findPosition(frame, draw=False)
        
        # Draw the dynamic menu
        frame = draw_menu(frame)

        # Process hand landmarks
        if landmark_list:
            x1, y1 = landmark_list[8][1], landmark_list[8][2]  # Index
            x2, y2 = landmark_list[12][1], landmark_list[12][2]  # Middle
            fingers = detector.fingerStatus()

            # Apply Kalman Filter to smooth positions
            x1_smooth, y1_smooth = kf_index.update(x1, y1)
            x2_smooth, y2_smooth = kf_middle.update(x2, y2)

            # Clear Canvas Gesture (All fingers open)
            if all(fingers):  # Check if all fingers are open
                if not clear_canvas_flag:  # Avoid clearing multiple times
                    image_canvas = np.zeros((720, 1280, 3), np.uint8)  # Clear canvas
                    clear_canvas_flag = True
                    cv2.putText(frame, 'Canvas Cleared!', (400, 400), 
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            else:
                clear_canvas_flag = False  # Reset flag when gesture ends

            # Color Selection Mode
            if fingers[1] and fingers[2]:
                xp, yp = 0, 0
                if y1_smooth < MENU_HEIGHT:  # Check if finger is in the menu area
                    menu_width = frame.shape[1] // len(COLORS)
                    selected_index = x1_smooth // menu_width
                    if 0 <= selected_index < len(COLORS):
                        draw_color = COLORS[selected_index]
                        cv2.putText(frame, f'Selected: {COLOR_NAMES[selected_index]}', 
                                   (900, 680), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                cv2.putText(frame, 'Color Select', (900, 680), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.line(frame, (x1_smooth, y1_smooth), (x2_smooth, y2_smooth), draw_color, 3)

            # Drawing Mode
            elif fingers[1] and not fingers[2]:
                if xp == 0 and yp == 0:
                    xp, yp = x1_smooth, y1_smooth
                
                thickness = ERASER_THICKNESS if draw_color == (0, 0, 0) else BRUSH_THICKNESS
                cv2.line(image_canvas, (xp, yp), (x1_smooth, y1_smooth), draw_color, thickness)
                xp, yp = x1_smooth, y1_smooth

                # Draw eraser ring if eraser is selected
                if draw_color == (0, 0, 0):
                    cv2.circle(frame, (x1_smooth, y1_smooth), ERASER_THICKNESS // 2, (0, 0, 255), 2)

        # Merge canvas and frame
        img_gray = cv2.cvtColor(image_canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        frame = cv2.bitwise_and(frame, cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR))
        frame = cv2.bitwise_or(frame, image_canvas)

        # Show FPS
        currentT = time.time()
        fps = int(1 / (currentT - previousT))
        cv2.putText(frame, f'FPS: {fps}', (10, 670), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        previousT = currentT
        
        cv2.imshow('Paint', frame)
        
        key = cv2.waitKey(1)
        if key == 27:  # ESC to exit
            break
        elif key == ord('f'):  # Toggle fullscreen
            cv2.setWindowProperty('Paint', cv2.WND_PROP_FULLSCREEN, 
                                not cv2.getWindowProperty('Paint', cv2.WND_PROP_FULLSCREEN))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
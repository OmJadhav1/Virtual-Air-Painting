import cv2
import numpy as np
import time
import os
import track_hands as TH

# Configuration
BRUSH_THICKNESS = 15
ERASER_THICKNESS = 100
COLORS = [
    (0, 0, 255),    # Red
    (47, 225, 245),  # Cyan
    (197, 47, 245),  # Pink
    (53, 245, 47),   # Green
    (0, 0, 0)       # Eraser (Black)
]
COLOR_NAMES = ["Red", "Cyan", "Pink", "Green", "Eraser"]
MENU_HEIGHT = 150  # Total height of the menu bar
MENU_SECTIONS = 2  # Color + Thickness sections
SECTION_HEIGHT = MENU_HEIGHT // MENU_SECTIONS  # Each section gets equal space
THICKNESS_OPTIONS = [10, 20, 30, 40]  # Predefined sizes

# Visual feedback colors
HIGHLIGHT_COLOR = (255, 255, 255)  # White for selected color
THICKNESS_HIGHLIGHT = (0, 255, 0)   # Green for selected thickness
PREVIEW_COLOR = (200, 200, 200)     # Gray for thickness preview
TEXT_COLOR = (255, 255, 255)       # White for text

# Kalman Filter Class


class KalmanFilter:
    def __init__(self, process_noise=1e-3, measurement_noise=1e-1):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                                             [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(
            2, dtype=np.float32) * measurement_noise

    def update(self, x, y):
        """Update the Kalman Filter with proper array conversion"""
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measurement)
        prediction = self.kf.predict()
        return int(prediction[0, 0]), int(prediction[1, 0])


def draw_menu(frame, active_color, active_thickness):
    """Draw improved menu with equal sections and better visual feedback"""
    # Draw color selection section (top half)
    color_width = frame.shape[1] // len(COLORS)
    for i, color in enumerate(COLORS):
        x1, y1 = i * color_width, 0
        x2, y2 = (i + 1) * color_width, SECTION_HEIGHT

        # Draw color button
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)

        # Add text label
        text_size = cv2.getTextSize(
            COLOR_NAMES[i], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x1 + (color_width - text_size[0]) // 2
        text_y = y1 + SECTION_HEIGHT - 20
        cv2.putText(frame, COLOR_NAMES[i], (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)

        # Highlight active color
        if color == active_color:
            cv2.rectangle(frame, (x1, y1), (x2, y2), HIGHLIGHT_COLOR, 3)

    # Draw thickness selection section (bottom half)
    thickness_width = frame.shape[1] // len(THICKNESS_OPTIONS)
    for i, thickness in enumerate(THICKNESS_OPTIONS):
        x1, y1 = i * thickness_width, SECTION_HEIGHT
        x2, y2 = (i + 1) * thickness_width, MENU_HEIGHT

        # Draw thickness preview circle
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        cv2.circle(frame, center, thickness // 2, PREVIEW_COLOR, -1)

        # Add size label
        cv2.putText(frame, str(thickness), (center[0] - 10, center[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Highlight active thickness
        if thickness == active_thickness:
            cv2.rectangle(frame, (x1, y1), (x2, y2), THICKNESS_HIGHLIGHT, 3)

    return frame


def main():
    # Initialization
    current_thickness = THICKNESS_OPTIONS[1]  # Default medium thickness
    draw_color = COLORS[0]
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

        # Draw the improved menu
        frame = draw_menu(frame, draw_color, current_thickness)

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
                    image_canvas = np.zeros(
                        (720, 1280, 3), np.uint8)  # Clear canvas
                    clear_canvas_flag = True
                    cv2.putText(frame, 'Canvas Cleared!', (400, 400),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            else:
                clear_canvas_flag = False  # Reset flag when gesture ends

            # Color/Thickness Selection Mode
            if fingers[1] and fingers[2]:  # Both index and middle fingers up
                xp, yp = 0, 0  # Reset drawing position

                # Check which section is being interacted with
                if y1_smooth < SECTION_HEIGHT:  # Color selection (top section)
                    selected_index = x1_smooth // (
                        frame.shape[1] // len(COLORS))
                    if 0 <= selected_index < len(COLORS):
                        draw_color = COLORS[selected_index]
                        # Visual feedback
                        cv2.putText(frame, f'Color: {COLOR_NAMES[selected_index]}',
                                    (900, 680), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                # Thickness selection (bottom section)
                elif SECTION_HEIGHT < y1_smooth < MENU_HEIGHT:
                    selected_index = x1_smooth // (
                        frame.shape[1] // len(THICKNESS_OPTIONS))
                    if 0 <= selected_index < len(THICKNESS_OPTIONS):
                        current_thickness = THICKNESS_OPTIONS[selected_index]
                        # Visual feedback
                        cv2.putText(frame, f'Size: {current_thickness}px',
                                    (900, 680), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                # Show selection mode indicator
                cv2.putText(frame, 'SELECT MODE', (900, 650),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Drawing Mode
            elif fingers[1] and not fingers[2]:
                thickness = ERASER_THICKNESS if draw_color == (
                    0, 0, 0) else current_thickness
                if xp == 0 and yp == 0:
                    xp, yp = x1_smooth, y1_smooth

                cv2.line(image_canvas, (xp, yp), (x1_smooth,
                         y1_smooth), draw_color, thickness)
                xp, yp = x1_smooth, y1_smooth

                # Draw eraser ring if eraser is selected
                if draw_color == (0, 0, 0):
                    cv2.circle(frame, (x1_smooth, y1_smooth),
                               ERASER_THICKNESS // 2, (0, 0, 255), 2)

        # Merge canvas and frame
        img_gray = cv2.cvtColor(image_canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        frame = cv2.bitwise_and(
            frame, cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR))
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

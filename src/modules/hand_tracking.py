import sys
import time
from threading import Thread

import cv2
import mediapipe as mp
import numpy as np


class HandDetector:
    def __init__(self, max_hands=2, detection_con=0.5, min_track_con=0.5, smoothing_factor=0.5):
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.min_track_con = min_track_con
        self.smoothing_factor = smoothing_factor
        self.prev_angle = None

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=False, max_num_hands=self.max_hands,
                                        min_detection_confidence=self.detection_con,
                                        min_tracking_confidence=self.min_track_con)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = np.array([4, 8, 12, 16, 20])

    def find_hands(self, img, flip_type=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        all_hands = []
        if results.multi_hand_landmarks:
            for handType, handLms in zip(results.multi_handedness, results.multi_hand_landmarks):
                my_hand = {}
                my_lm_list = np.array([[int(lm.x * img.shape[1]), int(lm.y * img.shape[0])] for lm in handLms.landmark])
                my_hand["lmList"] = my_lm_list
                my_hand["type"] = "Left" if handType.classification[0].label == "Right" else "Right" if flip_type else \
                    handType.classification[0].label
                all_hands.append(my_hand)
        return all_hands, img

    def get_index_finger_direction(self, my_hand):
        my_lm_list = my_hand["lmList"]
        if len(my_lm_list) >= 2:
            index_finger_base = my_lm_list[self.tipIds[1] - 1]
            index_finger_tip = my_lm_list[self.tipIds[1]]
            dx = index_finger_tip[0] - index_finger_base[0]
            dy = index_finger_tip[1] - index_finger_base[1]
            angle = np.arctan2(dy, dx)
            angle_degrees = np.degrees(angle)
            if angle_degrees < 0:
                angle_degrees += 360
            return angle_degrees
        else:
            return None

    def get_smoothed_angle(self, current_angle):
        if self.prev_angle is None:
            self.prev_angle = current_angle
        else:
            self.prev_angle = (self.smoothing_factor * current_angle) + ((1 - self.smoothing_factor) * self.prev_angle)
        return self.prev_angle

    def fingers_up(self, my_hand):
        my_lm_list = my_hand["lmList"]
        fingers = []
        if len(my_lm_list) >= 21:
            if my_lm_list[self.tipIds[0]][1] < my_lm_list[self.tipIds[0] - 1][1]:
                fingers.append(True)
            else:
                fingers.append(False)
            for finger_tip_id in range(1, 5):
                if my_lm_list[self.tipIds[finger_tip_id]][1] < my_lm_list[self.tipIds[finger_tip_id] - 2][1]:
                    fingers.append(True)
                else:
                    fingers.append(False)
        return fingers

    def draw_line_from_index_finder(self, img, my_hand):
        my_lm_list = my_hand["lmList"]
        if len(my_lm_list) >= 2:
            index_finger_tip = my_lm_list[self.tipIds[1]]
            index_finger_base = my_lm_list[self.tipIds[1] - 1]
            angle_degrees = self.get_index_finger_direction(my_hand)
            if angle_degrees is not None:
                smoothed_angle = self.get_smoothed_angle(angle_degrees)
                dx = np.cos(np.radians(smoothed_angle))
                dy = np.sin(np.radians(smoothed_angle))
                x1 = index_finger_tip[0] - 1000 * dx
                y1 = index_finger_tip[1] - 1000 * dy
                x2 = index_finger_tip[0] + 1000 * dx
                y2 = index_finger_tip[1] + 1000 * dy
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(img, f'{int(smoothed_angle)} deg', (index_finger_tip[0] - 50, index_finger_tip[1] + 30),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return img


def process_frames(detector, show_img):
    cap = cv2.VideoCapture(0)

    fps_counter = 0
    fps_timer = time.time()

    frame_time = 1.0 / 60  # 60 fps

    while True:
        start_time = time.time()
        success, img = cap.read()

        try:
            hands, img = detector.find_hands(img, flip_type=True)
        except (Exception, KeyboardInterrupt):
            sys.exit(1)

        for hand in hands:
            detector.draw_line_from_index_finder(img, hand)
            # fingers = detector.fingers_up(hand)
            # print(fingers)

        if show_img:
            cv2.imshow("Image", img)
            cv2.waitKey(1)

        end_time = time.time()
        elapsed_time = end_time - start_time

        if elapsed_time < frame_time:
            time.sleep(frame_time - elapsed_time)

        fps_counter += 1
        if time.time() - fps_timer >= 1:
            print(f"FPS: {fps_counter}")
            fps_counter = 0
            fps_timer = time.time()


def run(show_img: bool = True):
    detector = HandDetector(detection_con=0.8, max_hands=1)

    thread = Thread(target=process_frames, args=(detector, show_img,), daemon=True)
    thread.start()
    thread.join()


if __name__ == "__main__":
    run()

import sys
import time
from threading import Thread

import cv2
import mediapipe as mp


class PoseDetector:
    def __init__(self, detection_con=0.8, tracking_con=0.9):
        self.detection_con = detection_con
        self.tracking_con = tracking_con

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=self.detection_con,
                                      min_tracking_confidence=self.tracking_con, model_complexity=0)
        self.mp_draw = mp.solutions.drawing_utils

    def find_pose(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        return results.pose_landmarks, img

    def draw_pose(self, img, pose_landmarks):
        if pose_landmarks:
            self.mp_draw.draw_landmarks(img, pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                        landmark_drawing_spec=self.mp_draw.DrawingSpec(color=(255, 0, 0),
                                                                                       thickness=2,
                                                                                       circle_radius=2),
                                        connection_drawing_spec=self.mp_draw.DrawingSpec(color=(0, 255, 255),
                                                                                         thickness=2,
                                                                                         circle_radius=2))
        return img


def process_frames(detector, show_img):
    cap = cv2.VideoCapture(0)

    fps_counter = 0
    fps_timer = time.time()

    frame_time = 1.0 / 30  # 30 fps

    while True:
        start_time = time.time()
        success, img = cap.read()

        try:
            pose_landmarks, img = detector.find_pose(img)
        except (Exception, KeyboardInterrupt):
            sys.exit(1)

        img = detector.draw_pose(img, pose_landmarks)

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
    detector = PoseDetector(detection_con=0.8)

    thread = Thread(target=process_frames, args=(detector, show_img,), daemon=True)
    thread.start()
    thread.join()


if __name__ == "__main__":
    run()

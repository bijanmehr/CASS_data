import cv2
import mediapipe as mp
import sys
import os
from progress.bar import Bar
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def cord_parser(cord):
    if cord.pose_landmarks:
        pose_dict = {"nose": cord.pose_landmarks.landmark[0],
                    "left_eye_inner": cord.pose_landmarks.landmark[1],
                    "left_eye": cord.pose_landmarks.landmark[2],
                    "left_eye_outer": cord.pose_landmarks.landmark[3],
                    "right_eye_inner": cord.pose_landmarks.landmark[4],
                    "right_eye": cord.pose_landmarks.landmark[5],
                    "right_eye_outer": cord.pose_landmarks.landmark[6],
                    "left_ear": cord.pose_landmarks.landmark[7],
                    "right_ear": cord.pose_landmarks.landmark[8],
                    "mouth_left": cord.pose_landmarks.landmark[9],
                    "mouth_right": cord.pose_landmarks.landmark[10],
                    "left_shoulder": cord.pose_landmarks.landmark[11],
                    "right_shoulder": cord.pose_landmarks.landmark[12],
                    "left_elbow": cord.pose_landmarks.landmark[13],
                    "right_elbow": cord.pose_landmarks.landmark[14],
                    "left_wrist": cord.pose_landmarks.landmark[15],
                    "right_wrist": cord.pose_landmarks.landmark[16],
                    "left_pinky": cord.pose_landmarks.landmark[17],
                    "right_pinky": cord.pose_landmarks.landmark[18],
                    "left_index": cord.pose_landmarks.landmark[19],
                    "right_index": cord.pose_landmarks.landmark[20],
                    "left_thumb": cord.pose_landmarks.landmark[21],
                    "right_thumb": cord.pose_landmarks.landmark[22],
                    "left_hip": cord.pose_landmarks.landmark[23],
                    "right_hip": cord.pose_landmarks.landmark[24],
                    "left_knee": cord.pose_landmarks.landmark[25],
                    "right_knee": cord.pose_landmarks.landmark[26],
                    "left_ankle": cord.pose_landmarks.landmark[27],
                    "right_ankle": cord.pose_landmarks.landmark[28],
                    "left_heel": cord.pose_landmarks.landmark[29],
                    "right_heel": cord.pose_landmarks.landmark[30],
                    "left_foot_index": cord.pose_landmarks.landmark[31],
                    "right_foot_index": cord.pose_landmarks.landmark[32]
                    }
        return pose_dict

    else:
        return 0

if len(sys.argv) > 2:
    print('You have specified too many arguments')
    sys.exit()

if len(sys.argv) < 2:
    print('You need to specify the path to be listed')
    sys.exit()

input_path = sys.argv[1]

for entry in os.scandir(input_path):
    if not os.path.isfile(entry.path):
        if entry.name[:2].isnumeric():
            dir = os.path.join(entry.path,"start_wheel_cam2.avi")
            name = entry.name


            cap = cv2.VideoCapture(dir)
            success, image = cap.read()
            if cap.isOpened():
                width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            out_dir = "./"+name+"_pose_vid"+".mp4"
            writer = cv2.VideoWriter(out_dir, fourcc, fps, (width, height))

            bar = Bar('Processing Frames', max=length)

            frame_num = 0
            pose_data = {}
            with mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                min_detection_confidence=0.5) as pose:
                while cap.isOpened():
                    success, image = cap.read()
                    frame_num += 1
                    if not success:
                        break

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # To improve performance, optionally mark the image as not writeable to
                    # pass by reference.
                    image.flags.writeable = False
                    results = pose.process(image)
                    pose_data.update({frame_num: cord_parser(results)})
                    # Draw the pose annotation on the image.
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    # cv2.imshow('MediaPipe Pose', image)
                    writer.write(image)
                    bar.next()
                    if cv2.waitKey(5) & 0xFF == 27:
                        break

            np.save('./%s_pose_data.npy' %name, pose_data)

            bar.finish()

            cap.release()
            writer.release() 



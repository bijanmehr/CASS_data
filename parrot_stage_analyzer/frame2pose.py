import numpy as np
import cv2
from progress.bar import Bar
import os
import sys
import mediapipe as mp




if len(sys.argv) > 2:
    print('You have specified too many arguments')
    sys.exit()

if len(sys.argv) < 2:
    print('You need to specify the path to be listed')
    sys.exit()

input_path = sys.argv[1]

if not os.path.isdir(input_path):
    print('The path specified does not exist')
    sys.exit()



mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def poser(image):
  with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=2) as pose:
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_hight, image_width, _ = image.shape
    if results.pose_landmarks:
      annotated_image = image.copy()
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=results.pose_landmarks,
          connections=mp_pose.POSE_CONNECTIONS,
          landmark_drawing_spec=drawing_spec,
          connection_drawing_spec=drawing_spec)
      return [annotated_image, results]
    else:
      return [np.zeros(1),0]

def cord_parser(cord):
    if cord == 0 :
        return 0
    else:
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



for entry in os.scandir(input_path):
    print("working on ",entry.name)
    root_dir = os.path.join("./pose_data",entry.name[:-6] + "pose")
    os.mkdir(root_dir)
    pose_img_dir = os.path.join(root_dir,"pose_img")
    os.mkdir(pose_img_dir)

    length = len([name for name in os.listdir(entry.path) if os.path.isfile(os.path.join(entry.path, name))])
    bar = Bar('Processing Frames', max=length)
    pose_data = {}
    i = 0
    for pic in os.scandir(entry.path):
        if i % 15 == 0:
            frame = cv2.imread(pic.path)
            pose_img, pose_dict = poser(frame)
            pose_data.update({i: cord_parser(pose_dict)})
            if pose_img.any() == False:
                continue
            else:
                cv2.imwrite(os.path.join(pose_img_dir,"frame%d.jpg" %i), pose_img)
        i = i + 1
        bar.next()

    np.save(os.path.join(root_dir ,'%s_pose_data.npy'%entry.name), pose_data) 

    bar.finish()
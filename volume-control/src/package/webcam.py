import math
import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import numpy as np

# --- 1. SETUP MEDIAPIPE TASKS CONFIGURATION ---
# This finds the exact directory where THIS webcam.py file lives
# current_dir = os.path.dirname(os.path.abspath(__file__))
model_path="/mnt/data/Code/ai-projects/opencv_projects/volume-control/src/package/hand_landmarker.task"

# If 'hand_landmarker.task' is in the same folder as webcam.py:
# model_path = os.path.join(current_dir, 'hand_landmarker.task')
# Configured for standard IMAGE processing mode
options = vision.HandLandmarkerOptions( 
    base_options=python.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2
)


def frame_conversion(frame):
    
    rgb_matrix= cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_matrix)


MIN_DIST=30
MAX_DIST=180

def webcam():
    #opencv asks os to take the camera access, 0 means default camera
    
    cap=cv2.VideoCapture(0)
     # Track previous volume to prevent shell-command spamming
    last_volume = -1
    with vision.HandLandmarker.create_from_options(options) as landmarker:

    #run forever and capture the frames continuosly
        while True:
            #return success bool value and frame is numpy array
            success,frame=cap.read()
            # print(type(frame))
            #(480, 640, 3) mean 480 rows, 640 columns and 3 channels of color BGR
            # print(frame.shape)
            
            if not success:
                print("failed")
                break
            
            # print(f"Frame:{frame}")
            # timestamp=int(time.time() * 1000)
            mp_image=frame_conversion(frame)
            
            #send to model
            result=landmarker.detect(mp_image)
             # Print output if hands are found
            if result.hand_landmarks:
                hand=result.hand_landmarks[0]
                height, width, _ = frame.shape
                # 1. Index Finger Tip (Landmark 8)
                index_tip = hand[8]
                ix, iy = int(index_tip.x * width), int(index_tip.y * height)
                
                # 2. Thumb Tip (Landmark 4)
                thumb_tip = hand[4]
                tx, ty = int(thumb_tip.x * width), int(thumb_tip.y * height)

                distance = math.hypot(ix - tx, iy - ty)
                vol_percent=int(np.interp(distance,[MIN_DIST,MAX_DIST],[0,100]))
                # Map the distance to screen coordinates for the visual UI bar
                bar_y = int(np.interp(distance, [MIN_DIST, MAX_DIST], [400, 150]))
                # --- LINUX AUDIO SYSTEM CONTROL ---
                # Only execute shell updates if volume changes significantly (> 1% change)
                if abs(vol_percent - last_volume) > 1:
                    # Executes the background amixer instruction silently
                    os.system(f"amixer set Master {vol_percent}% > /dev/null 2>&1")
                    last_volume = vol_percent
                    print(f"Linux Audio Set To: {vol_percent}%")
                
                # --- OPENCV DRAWING INTERFACE ---
                cv2.circle(frame, (ix, iy), 10, (255, 0, 0), -1)  # Blue dot Index
                cv2.circle(frame, (tx, ty), 10, (0, 0, 255), -1)  # Red dot Thumb
                cv2.line(frame, (ix, iy), (tx, ty), (0, 255, 0), 2)  # Green Line
                
                # Draw Visual UI On Frame
                cv2.rectangle(frame, (50, 150), (85, 400), (255, 255, 255), 3)
                cv2.rectangle(frame, (50, bar_y), (85, 400), (0, 255, 0), -1)
                cv2.putText(frame, f"{vol_percent}%", (40, 430), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                # Print the pixel distance to your terminal
                print(f"Finger Distance: {int(distance)} pixels")
                print(f"Hand detected! Total hands found: {len(result.hand_landmarks)}")
            
            # print(f"RGB Frame: {rgb_frame}")
            #creates a new window with Webcam as title and showing frame continuosly
            cv2.imshow("Webcam",frame)
            
            if cv2.waitKey(12) & 0xFF==ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
    
    

import cv2

def frame_conversion(frame):
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    return frame

def webcam():
    #opencv asks os to take the camera access, 0 means default camera
    cap=cv2.VideoCapture(0)
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
        rgb_frame=frame_conversion(frame)
        # print(f"RGB Frame: {rgb_frame}")
        #creates a new window with Webcam as title and showing frame continuosly
        cv2.imshow("Webcam",frame)
        
        if cv2.waitKey(12) & 0xFF==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
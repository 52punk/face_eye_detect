
#Importing OpenCV

import cv2

framewidth = 640      # Width and height of the video captured
frameheight = 480

#To start video capturing 0-> from webcam

cap = cv2.VideoCapture(0)

cap.set(3, framewidth)
cap.set(4, frameheight)
cap.set(10, 150)          # Brightness

fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter('face_and_eye.avi', fourcc, 20.0, (640,480))

#Importing the haarcascade models
#these files can be downloaded from https://github.com/opencv/opencv/tree/master/data/haarcascades

face_Cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_Cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

#Defining the font of the text which we will be using for labelling

font = cv2.FONT_HERSHEY_SIMPLEX

#Running the while loop and reading all the frames from the video fro webcam
#using frame 

while True:
    ret, frame = cap.read()
    
    #Flipping the frame for preventing the mirror frame
    
    frame = cv2.flip(frame, 1)
    
    #Changing the frame from BGR to GRAY
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Using the object linked to haarcascade_frontalface_default.xml to detect faces
    
    faces = face_Cascade.detectMultiScale(frame_gray, 1.3, 5)
    
    #This object returns the points from the frame where the face is located
    
    for(x, y, w, h) in faces:
        
        #Making a rectangular box around the face
        
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        
        #Putting the text on the box of face
        
        frame = cv2.putText(frame, "Face", (x,y), font, 1, (0,255,255), 2, cv2.LINE_AA)
        
        #Separating our RegionOfInterest (roi) in gray format for eye detection
        
        roi_gray = frame_gray[y:y+h, x:x+w]
        
        #roi in color format for the output purpose
        
        roi_color = frame[y:y+h, x:x+w]
        
        #Using the object linked to haarcascade_eye.xml to detect eyes
        
        eyes = eye_Cascade.detectMultiScale(roi_gray)
        
        #This object returns the points from roi_gray where the eyes are located
        
        for(ex, ey, ew, eh) in eyes:
            
            #Same operation as for face
            
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            frame = cv2.putText(frame, "Eyes", (x+ex,y+ey), font, 0.5, (0,255,255), 2, cv2.LINE_AA)
    
    if ret == True:
        
        out.write(frame)
    
    
    #Giving the output frame
    
    cv2.imshow("frame", frame)
    
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
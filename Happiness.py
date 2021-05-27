#!/usr/bin/env python
# coding: utf-8

# ## Making a model that detects face realtime

# In[1]:


import cv2


# Importing the cascades

# In[6]:


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")


# Defining a function that detects face

# In[7]:


def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (x1, y1, w1, h1) in eyes:
            cv2.rectangle(roi_color, (x1,y1), (x1+w1, y1+h1), (255,0,0), 1)
            
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
            
    return frame
        


# Opening the webcam

# In[8]:


video_capture = cv2.VideoCapture(0)


# In[9]:


while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
        
video_capture.release()
cv2.destroyAllWindows()


# In[ ]:





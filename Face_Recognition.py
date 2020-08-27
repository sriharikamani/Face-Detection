#!/usr/bin/env python
# coding: utf-8

# ## Real Time Face Detection
# Computer Vision: Real Time live image capture, detect image from the trained dataset and show basic demographic details with name and emotion.

# In[44]:


import pandas as pd
import numpy as np
import warnings  
import os
import pendulum
warnings.filterwarnings('ignore')
import tkinter as tkhttp 
import cv2
import re
from os import listdir
from os.path import isfile, join
from keras.preprocessing import image
from tensorflow.keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file


# ### Load Emotion Model

# In[46]:


path = 'C:/Users/sriha/Desktop/'

modelEmo = model_from_json(open(path + "Emotion_Analysis.json", "r").read())
modelEmo.load_weights(path + "Emotion_Analysis.h5")


# In[45]:


CASE_PATH = path + 'haarcascade_frontalface_default.xml'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'


# ### Load known face model which was created with different face images

# In[47]:


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(path + "data_model.yml")


# ### Routine to detect the image

# In[48]:


def face_detector(img, size = 0.5):
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)          # Convert from RGB to GRAY
    
    #Search the co-ordinates of the image
    faces = face_classifier.detectMultiScale(gray,scaleFactor=1.3,minNeighbors =5) 

    if faces is():         # No faces / No-Match
        return img,[]
    
    for(x,y,w,h) in faces: # Match
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)   
     
    # Get face and resize
        face_img  = img[y:y+h, x:x+w]
        face_img  = cv2.resize(face_img, (48,48))

    return img,face_img


# ### Routine to fetch the user name with face emotion

# In[49]:


def get_user_name(id,face):
     
    temp = str(id)
    temp = int(temp[:3])
    ################################################# 
    # Read CSV file to fetch the name of the employee 
    #################################################    
     
    data = pd.DataFrame()
    os.chdir(path)
    data = pd.read_csv("employee.csv")
    data.drop('Unnamed: 0', axis = 1 , inplace= True)
    data= data.sort_values('ID')
    
    # Fetch employee name

    name = data.loc[data['ID'] == temp]
    name = name['Name'].iloc[0]
    
    ######################### 
    # Get the face emotion 
    #########################
     
    gray_img       = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    faces_detected = face_classifier.detectMultiScale(gray_img, 1.32, 5)
    
    for (x,y,w,h) in faces_detected:
        
        #cv2.rectangle(face,(x,y),(x+w,y+h),(255,0,0))
        face_img    = gray_img[y:y+w,x:x+h]      #cropping region of interest i.e. face area from  image
        face_img    = cv2.resize(face_img,(48,48))
        img_pixels  = image.img_to_array(face_img)
        img_pixels  = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255
        pred        = modelEmo.predict(np.array(img_pixels))
        index       = np.argmax(pred[0])
        
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        pred_emo = emotions[index]
        
        return name, temp, pred_emo , x, y
    


# ### Predict the unknown live image

# In[50]:


face_classifier = cv2.CascadeClassifier(CASE_PATH)

video = cv2.VideoCapture(0)  # Read the image from camera
match = 'N'

while True:
     
    ret, img_frame  = video.read()  # captures frame and returns boolean value and captured image
    image_, face    = face_detector(img_frame)
             
    try:
        # Reading the image as gray scale image
        face   = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = recognizer.predict(face)
        
        if result[1] < 500:
            confidence          = int(100*(1-(result[1])/300))
            user,empid,emo,h,w  = get_user_name(str(result[0]),img_frame)
            display_string      = user 
        
        cv2.putText(image_,display_string,(2,50), cv2.FONT_HERSHEY_SIMPLEX,1,(150,100,200),2)
        cv2.putText(image_,'EmpId=' + str(empid),(2,85), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
        cv2.putText(image_,pendulum.now().to_day_datetime_string(),(370,80), cv2.FONT_HERSHEY_PLAIN,1,(0, 0, 0),2)
        
        cv2.putText(image_,emo,(int(h)+10,int(w)-15), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        
        resized_img = cv2.resize(image_, (1600, 960))
        cv2.imshow('Face Cropper', resized_img)
        match = 'Y'
    
    except:
        cv2.putText(image_, "Face Not Found", (250, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        resized_img = cv2.resize(image_, (1600, 960))
        cv2.imshow('Face Cropper', resized_img)
        match = 'N'
        #img_frame = ''
        pass

    if (cv2.waitKey(1)==13):   # 13 is ASCII code for Enter Key
        break

video.release()
cv2.destroyAllWindows()

## End of the Program
# In[ ]:





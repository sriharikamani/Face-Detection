#!/usr/bin/env python
# coding: utf-8

# ## Face Detection - Build Data for new user
# Computer Vision: Real Time live image capture with demographic details The purpose of this python code is to capture basic demographic details and capture 50 faces of each user and then build the model

# In[1]:


import pandas as pd
import numpy as np
import warnings  
import os
 
warnings.filterwarnings('ignore')
import tkinter as tk
import cv2
import numpy as np

import tensorflow as tf
from tensorflow import keras

from os import listdir
from os.path import isfile, join
from keras.preprocessing import image
from tensorflow.keras.models import model_from_json


# ### Load Emotion Model

# In[2]:


path = 'C:/Users/sriha/Desktop/fr/'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'


# ### Routine to capture face features

# In[3]:


face_classifier = cv2.CascadeClassifier(path + 'haarcascade_frontalface_default.xml')

def face_extractor(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)          # Convert from RGB to GRAY
    
    #Search the co-ordinates of the image
    faces = face_classifier.detectMultiScale(gray,scaleFactor=1.3,minNeighbors =5) 

    if faces is():       # No faces
        return None

    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]
        
    return cropped_face


# ### Prepare the dataset 

# In[4]:


#os.remove('employee.csv')


# #### Phase 1:  Capature basic demographic details such as Name, Gender and ID

# In[9]:


def save_entry_fields():
    
    NAME_.append(e1.get())  
    NAME_.append(e2.get())

def combine_funcs(*funcs):
    def combined_func(*args, **kwargs):
        for f in funcs:
            f(*args, **kwargs)
    return combined_func

def add_gender():
    
    temp = selected.get()
    
    if (temp == 1):
        NAME_.append('Male')
    else:
        NAME_.append('Female')
        
        
window = tk.Tk()
window.title("Welcome to Employee Add Menu")
window.geometry('550x250')
NAME_   = []
GENDER_ = []

tk.Label(window, text="Name",fg="blue",font="lucida 10 bold").grid(row=0)
tk.Label(window, text="    ID",fg="blue",font="lucida 10 bold").grid(row=1)
tk.Label(window, text="Gender   ",fg="blue",font="lucida 10 bold").grid(row=2)

e1 = tk.Entry(window)
e2 = tk.Entry(window)

selected = tk.IntVar()
selected.set(2)
e3 = tk.Radiobutton(window,text='Male', variable = selected,value=1,command = add_gender)
e4 = tk.Radiobutton(window,text='Female', variable = selected,value=2,command = add_gender)


e1.grid(row=0, column=1)
e2.grid(row=1, column=1)
e3.grid(row=2, column=1)
e4.grid(row=2, column=2)

tk.Button(window, 
          text='Quit',bg="blue", fg="white",font=("Arial Bold", 13),
          command=window.destroy).grid(row=13,
                                    column=1, 
                                    sticky=tk.W, 
                                    pady=4)

tk.Button(window, 
          text='Submit', bg="blue", fg="white",font=("Arial Bold", 13),
                         command=combine_funcs(save_entry_fields,window.destroy)).grid(row=13, 
                                                       column=2, 
                                                       sticky=tk.W, 
                                                       pady=4)
  

tk.mainloop()

Gender   = pd.DataFrame([NAME_[0]],columns= ['Gender'])
Name     = pd.DataFrame([NAME_[1]],columns= ['Name'])
ID       = pd.DataFrame([NAME_[2]],columns= ['ID'])
df       = pd.concat([ID,Name,Gender],axis=1)

################
# Append to CSV
################

#df.to_csv('employee.csv',mode='a', header=True)
df.to_csv('employee.csv', mode='a', header=False) 


ID_     = NAME_[2]

#del df
#del data


# #### Phase  2:   Capture and save the image (50 different faces)

# In[ ]:


# Save face images and show the images

video = cv2.VideoCapture(0)  # 0 to specify to use built-in camera
count = 0

while True:
    ret, frame = video.read()
    if face_extractor(frame) is not None:
        count+=1
        face = cv2.resize(face_extractor(frame),(200,200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        file_name_path = path + 'Images/'+ID_ + '_'+ str(count)+'.jpg'   # Save images in .jpg format
        cv2.imwrite(file_name_path,face)
        cv2.putText(face, str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper',face)
    else:
        #print("Face not Found")
        pass

    if cv2.waitKey(1)==13 or count==50: # 13 is ASCII code for Enter Key or Maximum of 50 images
        break

video.release()
cv2.destroyAllWindows()
print('Collecting Samples Completed...........')


# #### Phase 3: Create and Train the model  by assigning label starting from 1 to 50 with ID as prefix for each encoded face

# In[ ]:


imgs_path = path + 'Images/'
Training_Data, Labels, onlylabels = [], [] , []

onlylabels = [f for f in listdir(imgs_path) if isfile(join(imgs_path ,f))]

for i, files in enumerate(onlylabels):
    image_path = imgs_path + onlylabels[i]
    images     = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(onlylabels[i])

# Extract only numerics from string
Labels = list(map(lambda sub:int(''.join( 
          [ele for ele in sub if ele.isnumeric()])), Labels)) 

Labels = np.asarray(Labels, dtype=np.int32)
model  = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_Data), np.asarray(Labels))
 
print("Model Training Complete!!!!!")


# ### Save the model

# In[ ]:


model.write(path + "data_model.yml")

 ## End of the Program
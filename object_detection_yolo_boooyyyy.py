#!/usr/bin/env python
# coding: utf-8

# In[2]:

import contextlib
import os
from multiprocessing import freeze_support

import matplotlib as plt
from os.path import join as pjoin
from ultralytics import YOLO

# Load a model


# In[3]:






if __name__ == '__main__':
    freeze_support()
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data="C:\\Users\\egors\\PycharmProjects\\object_detection\\Codev.v3i.yolov8\\data.yaml", epochs=20,
                workers=4, device=0)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    results = model.predict(source='C:\\Users\\egors\\PycharmProjects\\object_detection\\Codev.v3i.yolov8\\test\\images',save=True)
    success = model.export(format="onnx")  # export the model to ONNX format


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:

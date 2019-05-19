# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 17:16:01 2019

@author: ragga
SPLIT Data into training test and validation sets
"""

import os
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

# Load the data that needs to be analyzed
#aud_file_path = 'C:/Carnegie Mellon/10707-DL/Project/joint_model/audio_samples/'
#vid_file_path = 'C:/Carnegie Mellon/10707-DL/Project/joint_model/extracted_features_video/'

# Path where we want to create the datasets
# The directory structure should look like:
    # audio_train
        # 0
        # 1
        # ...
        # 7
    # audio_test 
        # 0
        # 1
        # ...
        # 7
    # audio_val
        # 0
        # 1
        # ...
        # 7
    # Similarly for video set

folders = ['0', '1', '2', '3', '4', '5', '6', '7']
def create_test_train_val_sets(tmpdir):
    vid_file_path = tmpdir + '/' + 'vid_features/'
    aud_file_path = tmpdir + '/' + 'aud_features/'
    aud_train_set = tmpdir + '/' + 'aud_train_tmp/'
    aud_val_set = tmpdir + '/' + 'aud_val_tmp/'
    aud_test_set = tmpdir + '/' + 'aud_test_tmp/'
    vid_train_set = tmpdir + '/' + 'vid_train/'
    vid_val_set = tmpdir + '/' + 'vid_val/'
    vid_test_set = tmpdir + '/' + 'vid_test/'

    created = False
    paths = [vid_train_set, vid_val_set, vid_test_set, aud_train_set, aud_val_set, aud_test_set]
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)
            created = True
            for i in folders:
                subfolder = path + i
                os.mkdir(subfolder)
    
    if not created:
        return
    
    image_data = []
    aud_data = []
    img_files = []
    aud_files = []

    for file in os.listdir(vid_file_path):
        image = cv2.imread(os.path.join(vid_file_path, file));
        image_data.append(image)
        img_files.append(file)
#
    for file in os.listdir(aud_file_path):
        aud = np.load(os.path.join(aud_file_path, file))
        aud_data.append(aud)
        aud_files.append(file)


    img_train, img_test, img_train_files, img_test_files = train_test_split(image_data, img_files, test_size=0.5, random_state=42)
    aud_train, aud_test, aud_train_files, aud_test_files = train_test_split(aud_data, aud_files, test_size=0.5, random_state=42)
    img_val, img_test, img_val_files, img_test_files = train_test_split(img_test, img_test_files, test_size=0.5, random_state=42)
    aud_val, aud_test, aud_val_files, aud_test_files = train_test_split(aud_test, aud_test_files, test_size=0.5, random_state=42)

    print("starting to write video data")

    for i in range(len(img_train)):
        img = img_train[i]
        fp = vid_train_set + str(int(img_train_files[i][7:8]) - 1) + '/'
        cv2.imwrite(fp + "frame%d.jpg" %i, img)
        
    for i in range(len(img_test)):
        img = img_test[i]
        fp = vid_test_set + str(int(img_test_files[i][7:8]) - 1) + '/'
        cv2.imwrite(fp + "frame%d.jpg" %i, img)
    
    for i in range(len(img_val)):
        img = img_val[i]
        fp = vid_val_set + str(int(img_val_files[i][7:8]) - 1) + '/'
        cv2.imwrite(fp + "frame%d.jpg" %i, img)    

    print("starting to write audio data")

    for i in range(len(aud_train)):
        aud = aud_train[i]
        aud = np.reshape(aud, (1, 40))
        print(aud.shape)
        fp = aud_train_set + str(int(aud_train_files[i][7:8]) - 1) + '/'
        print(fp)
        np.save(fp + "aud%d" %i, aud)

    for i in range(len(aud_test)):
        aud = aud_test[i]
        aud = np.reshape(aud, (1, 40))
        fp = aud_test_set + str(int(aud_test_files[i][7:8]) - 1) + '/'
        np.save(fp + "aud%d" %i, aud)
#   
    for i in range(len(aud_val)):
        aud = aud_val[i]
        aud = np.reshape(aud, (1, 40))
        fp = aud_val_set + str(int(aud_val_files[i][7:8]) - 1) + '/'
        print('VAL FILE', fp)
        np.save(fp + "aud%d" %i, aud)
    


# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 22:51:40 2019

@author: ragga
split audio data and video data, organize it into folders
"""

import os
import librosa
import numpy as np
import cv2

#the audio path should have both speech and song datasets
#video_path = 'C:/Carnegie Mellon/10707-DL/Project/Dataset'
#the video set should have the speech dataset
#audio_path = 'C:/Carnegie Mellon/10707-DL/Project/Audio_dataset'

#video_store_path = 'C:/Carnegie Mellon/10707-DL/Project/joint_model/extracted_features_video/'
#audio_store_path = 'C:/Carnegie Mellon/10707-DL/Project/joint_model/extracted_features_audio_full/'

def preprocess(audio_path, video_path, tmpdir):
    video_store_path = tmpdir + '/' + 'vid_features/'
    audio_store_path = tmpdir + '/' + 'aud_features/'
    make_dir_aud = False
    make_dir_vid = False
    
    if not os.path.exists(video_store_path):
        os.mkdir(video_store_path)
        make_dir_vid = True
    if not os.path.exists(audio_store_path):
        os.mkdir(audio_store_path)
        make_dir_aud = True
    
    if (not make_dir_aud and not make_dir_vid):
        return
        
    for subdir, dirs, files in os.walk(audio_path):
        for file in files:
            try:
                X, sample_rate = librosa.load(os.path.join(subdir,file), res_type='kaiser_best')
                mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) 
                mfccs = np.asarray(mfccs)
                np.save(audio_store_path + file[0:-4], mfccs)
            except ValueError:
                continue  

    for subdir, dirs, files in os.walk(video_path):
        for file in files:     
            if file[0:2] == '02': # use only video files without audio
                file_path = os.path.join(subdir, file)
                vidcap = cv2.VideoCapture(file_path)
                images = []
                success, image = vidcap.read()
                count = 0
                while success:
                    vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 250))
                    image = image[150:662, 400:912]                  # crop the image  
                    image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)        # resize the image
                    images.append(image)
                    success, image = vidcap.read()
                    count += 1
            
              # we want to create a collage of 512x512 images stacked horizontally
              # horizontal collage
                numPairs = len(images)//2;
                for i in range(numPairs):
                    collage = np.hstack([images[2*i], images[2*i + 1]])
                    cv2.imwrite(video_store_path + file[0:-4] + "-frame%d.jpg" %i + ".jpg", collage)
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 16:30:12 2019

@author: ragga
Duplicate the audio signals to match the number of values in image folders
"""

import os
import numpy as np
import math

## This path is the path to the currect dataset
folders = ['0', '1', '2', '3', '4', '5', '6', '7']

def expand_audio(tmpdir):
    curr_aud_train_path = tmpdir + '/' + 'aud_train_tmp'
    curr_aud_val_path = tmpdir + '/' + 'aud_val_tmp'
    curr_aud_test_path = tmpdir + '/' + 'aud_test_tmp'
    video_train_path = tmpdir + '/' + 'vid_train/'
    video_val_path = tmpdir + '/' + 'vid_val/'
    video_test_path = tmpdir + '/' + 'vid_test/'
    
    final_aud_train_path = tmpdir + '/' + 'aud_train'
    final_aud_test_path = tmpdir + '/' + 'aud_test'
    final_aud_val_path = tmpdir + '/' + 'aud_val'
    
    vid_paths = [video_train_path, video_test_path, video_val_path]
    aud_paths = [curr_aud_train_path, curr_aud_test_path, curr_aud_val_path]
    final_aud_paths = [final_aud_train_path, final_aud_test_path, final_aud_val_path]
    
    created = False
    for path in final_aud_paths:
        if not os.path.exists(path):
            created = True
            os.mkdir(path)
            for i in folders:
                subfolder = path + '/' + i
                os.mkdir(subfolder)
    
    if not created:
        return

    for p in range(len(vid_paths)):
        # These are the number of files each set contains
        path = vid_paths[p]
        output_sizes = []
        for subdir, dirs, files in os.walk(path):
            fcount = 0
            for file in files:
                fcount += 1
            if fcount != 0:
                output_sizes.append(fcount)
        #print("Output sizes ", len(output_sizes))
        curr_aud_path = aud_paths[p]
        for i in range(len(folders)):
            files = []
            count = 0
            aud_path = curr_aud_path + '/' + folders[i] + '/'
            for file in os.listdir(aud_path):
                data = np.load(os.path.join(aud_path, file))
                files.append(data)
                count += 1
            
            #print("Num of files in audio folder " , len(files))
            vid_count = output_sizes[i]
            #print("Number of video files", vid_count)
            num_iter = math.ceil(vid_count/count)
            cat_files = []
            for j in range(num_iter):
                cat_files += files
            save_path = final_aud_paths[p] + '/' + folders[i] + '/'
            for k in range(vid_count):
                np.save(save_path + "aud%d" %k, cat_files[k])
        
            
            
    


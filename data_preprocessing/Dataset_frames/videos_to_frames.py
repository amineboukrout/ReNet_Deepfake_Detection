import numpy as np
import cv2
import sys
import time
import os
import csv
import face_recognition
import glob
from subprocess import call

def get_video_parts(vid_path):
    """"let's say path is Data\Deepfake\0.mp4
    class = Deepfake
    file_name = 0
    file_ext = mp4"""
    parts = str(vid_path).split(os.path.sep)
    class_ = parts[-2]
    file_name = str(parts[-1]).split('.')[0]
    return class_, file_name

def check_already_extracted(dataset, class_, video_name):
    """Check to see if we created the -0001 frame of this file."""
    print(os.path.join(dataset, class_, video_name))
    return bool(os.path.exists(os.path.join(dataset, class_, video_name)))

def get_nb_frames_for_video(dataset, video_parts):
    """Given video parts of an (assumed) already extracted video, return
    the number of frames that were extracted."""
    classname, filename_no_ext = video_parts
    # print(os.path.join(dataset, classname,
    #                             filename_no_ext, '*.jpg'))
    generated_files = glob.glob(os.path.join(dataset, classname,
                                             filename_no_ext, '*.jpg'))
    return len(generated_files)

def extract_frames_to_images(dataset_name='Dataset', new_dataset_name='Dataset_frames'):
    data_file = []
    folders = [dataset_name + os.path.sep + 'Real' + os.path.sep, dataset_name + os.path.sep + 'Deepfake' + os.path.sep]
    # folders = [dataset_name + os.path.sep + 'Deepfake' + os.path.sep]

    # some folder creation
    # print(os.path.exists('dataset_pre'))
    if not os.path.exists(new_dataset_name):
        os.mkdir(new_dataset_name)
    if not os.path.exists(new_dataset_name + os.path.sep + 'Deepfake'):
        os.mkdir(new_dataset_name + os.path.sep + 'Deepfake')
    if not os.path.exists(new_dataset_name + os.path.sep + 'Real'):
        os.mkdir(new_dataset_name + os.path.sep + 'Real')

    for folder in folders:
        video_paths = glob.glob(os.path.join(folder, '*.mp4'))

        for video_path in video_paths:
            video_parts = get_video_parts(video_path)
            class_, file_name = video_parts

            if not check_already_extracted(new_dataset_name, class_, file_name):
                # print(os.path.exists(os.path.join('Dataset',class_,file_name+'.mp4')))
                # os.mkdir(os.path.join(new_dataset_name, class_, file_name))
                src = os.path.join(os.getcwd(), video_path)
                dest_folder = os.path.join(os.getcwd(), new_dataset_name, class_, file_name)
                os.mkdir(dest_folder)
                dest_folder_files = os.path.join(os.getcwd(), dest_folder, file_name + '-%04d.jpg')
                # print(dest_folder_files)

                # call([
                #     "C:\\Users\\Amine_Boukrout\\PycharmProjects\\ffmpeg-20191122-89aa134-win64-static\\bin\\ffmpeg",
                #     "-i", src, dest_folder_files])

                call([
                    "/local/java/ffmpeg/ffmpeg", "-i", src, dest_folder_files
                ])

            no_of_frames = get_nb_frames_for_video(new_dataset_name, video_parts)
            data_file.append([class_, file_name, no_of_frames])
            print('Video file {} generated {} files of frames'.format(file_name, no_of_frames))

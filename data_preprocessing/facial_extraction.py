import numpy as np
import cv2
import sys
import time
import os
import csv
import face_recognition
import glob

from subprocess import call
from moviepy.video.io.VideoFileClip import VideoFileClip
# from lstm_processor import process_image_path, process_image_arr

# facial extraction
def largest_face_size(vid_file):
    largest_face_size_H = 0
    largest_face_size_W = 0
    count = 0
    while True:
        ret, frame = vid_file.read()  # this extracts the next fram

        # if there are dataset_pre_images more frames, i.e. ret==False, than quit
        if not ret: break

        # rgb_frame = frame[:, :, ::-1] # BGR (OpenCV) to RGB (face_recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        locations_of_faces = face_recognition.face_locations(rgb_frame)
        if locations_of_faces:
            top_i, right_i, bottom_i, left_i = locations_of_faces[0]

            height_i = bottom_i - top_i
            width_i = right_i - left_i
            if height_i > largest_face_size_H:
                largest_face_size_H = height_i
            if width_i > largest_face_size_W:
                largest_face_size_W = width_i
        count += 1
    return largest_face_size_W, largest_face_size_H

def vid_init(video_file):
    vid = cv2.VideoCapture(video_file)
    return vid

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

# get frame values; i.e. number of frames in videos
def extract_face(dataset_name='Dataset_video_splited', new_dataset_name='Dataset_Face_Extracted', new_dataset_name0='Dataset_Face_Extracted_picS', bbox_bias=30, bbox_size=299,
                 frames=15, extract_as_imgs = False):
    output_folder = new_dataset_name
    data_file = []
    folders = [dataset_name + os.path.sep + 'Real' + os.path.sep, dataset_name + os.path.sep + 'Deepfake' + os.path.sep]
    # folders = [dataset_name + os.path.sep + 'Deepfake' + os.path.sep]

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
            no_frames_to_be_written = 0
            print(video_path)
            input_video = vid_init(video_path)
            v, m = input_video.read()
            length = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            frame_number = 0
            count = 0
            frame_list = []
            largest_face_width, largest_face_height = largest_face_size(input_video)
            print('largest_face_height: {}'.format(largest_face_height))
            print('largest_face_width: {}'.format(largest_face_width))
            input_video.release()
            input_video = vid_init(video_path)
            print('before while')

            while True:
                ret, frame = input_video.read()
                print(ret)
                frame_number += 1
                if not ret: break
                rgb_frame = frame #cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)

                extract_as_imgs = True #extract_as_imgs
                if extract_as_imgs:
                    video_partso = get_video_parts(video_path)
                    class_, file_name = video_parts

                    if not check_already_extracted(new_dataset_name, class_, file_name):
                        # print(os.path.exists(os.path.join('Dataset',class_,file_name+'.mp4')))
                        # os.mkdir(os.path.join(new_dataset_name, class_, file_name))
                        src = video_path
                        dest_folder = os.path.join(new_dataset_name, class_, file_name)
                        os.mkdir(dest_folder)
                        dest_folder_files = os.path.join(dest_folder, file_name + '-%04d.jpg')
                        call([
                            "/local/java/ffmpeg/ffmpeg",
                            "-i", src, dest_folder_files])

                    no_of_frames = get_nb_frames_for_video(new_dataset_name, video_parts)
                    data_file.append([class_, file_name, no_of_frames])
                    print('Video file %s generated %d files of frames'.format(file_name, no_of_frames))

                print('face locations')
                print(face_locations)
                if face_locations:
                    print('A face was found in frame no. {} of video {}'.format(frame_number, video_path))
                    top, right, bottom, left = face_locations[0]

                    # obtain largest width
                    if (right - left) < largest_face_width: right = right + (largest_face_width - (right - left))

                    # obtain largest height
                    if (bottom - top) < largest_face_height: bottom = bottom + (largest_face_height - (bottom - top))

                    frame = frame[top - bbox_bias:bottom + bbox_bias, left - bbox_bias:right + bbox_bias]

                    try:
                        frame = cv2.resize(frame, (bbox_size, bbox_size), interpolation=cv2.INTER_LINEAR)
                        frame_list = frame_list + [frame]
                    except Exception as e:
                        print(str(e))
                else:
                    print('Frame {} is missing a face in video {}'.format(frame_number, video_path))

                # we can save the frames as images but this will be tedious
                count += 1

            # writing the frames with faces to a video file
            if len(frame_list) >= frames:
                # initialize the output video
                output_video = cv2.VideoWriter(output_folder + os.path.sep + class_ + os.path.sep + file_name + '.mp4',
                                               fourcc, length, (bbox_size, bbox_size))

                # creating the video
                for f in range(len(frame_list)):
                    print('Writing frame {} of {}'.format(f + 1, length))
                    output_video.write(frame_list[f])
            else:
                if len(frame_list) > 1:  # (frames*0.75):
                    # initialize the output video
                    output_video = cv2.VideoWriter(
                        output_folder + os.path.sep + class_ + os.path.sep + file_name + '.mp4', fourcc, length,
                        (bbox_size, bbox_size))

                    print('Duplicating frames for video {}'.format(video_path))
                    print(len(frame_list))
                    frame_list = frame_list + [frame_list[0]] * (frames - len(frame_list))
                    for f in range(len(frame_list)):
                        print('Writing frame {} of {}'.format(f + 1, length))
                        output_video.write(frame_list[f])
                else:
                    print('Discarding invalid video {}'.format(file_name))
                    # import time
                    # time.sleep(16)

            input_video.release()
            cv2.destroyAllWindows()
            print('Done with a video!')

    with open('data_info.csv', 'w') as out:
        writer = csv.writer(out)
        writer.writerows(data_file)

    print("Extracted and wrote %d video files." % (len(data_file)))

# extract_face(str(os.getcwd())[:-5]+'\\vids\\', str(os.getcwd())[:-5]+'\\output\\', 20, 100, 20)
# extract_face('Dataset','DFE')
extract_face()
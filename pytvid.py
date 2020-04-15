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
from lstm_processor import process_image_path, process_image_arr

# abspath = os.path.abspath('C:\\Users\Amine_Boukrout\PycharmProjects\\videdit\\pytvid.py')
# dname = os.path.dirname(abspath)
# os.chdir(dname)
# print(os.getcwd())
# sys.exit()


# print(str(os.getcwd())[:-5])
# print(os.path.sep)
# sys.exit()
# cap = cv2.VideoCapture()
# file_path = str(os.getcwd())[:-5]+'\\vids\\30.mp4'

def clear_folder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
    print('{} cleared'.format(folder))


def vid_init(video_file):
    vid = cv2.VideoCapture(video_file)
    return vid


# cap = vid_init(file_path)
# cap = cv2.VideoCapture(r'C:\\Users\\Amine_Boukrout\\Documents\\masters\\MSc_Warwick\\Deepfakes\\code\\thirdeye-master\\Data\\TEST\\DF_RAW\\111.mp4')

# retrieving data
# rin = False
# if rin:
#     data = []
#     vid = []
#     while True:
#         ret, frame = cap.read()
#         # print('ret is ', ret)
#         if not ret: break
#         vid.append(frame)
#     vid = np.array(vid, dtype=np.float32)
#     data.append(vid)
#     cap.release()


# ----------------------------------------------------------------------------------------------------------------------

def retrieve_videos_as_frames(path):
    video_frames_all = []
    videos_folder = os.listdir(path)
    for video_name in videos_folder:
        print(video_name)
        video_input = cv2.VideoCapture(path + video_name)
        video_input_frames = []
        while True:
            ret, frame = video_input.read()
            print(ret)
            if not ret: break
            video_input_frames.append(frame)
        video_input.release()
        video_input_frames = np.array(video_input_frames, dtype=np.float32)
        video_frames_all.append(video_input_frames)
        break
    return video_frames_all

# ----------------------------------------------------------------------------------------------------------------------
# data = retrieve_videos_as_frames(os.getcwd()[:-5]+'\\vids\\')
# print(len(data))

# get number of frames in video
# def get_nb_frames_for_video(video_name):
#     """Given video name, return the number of frames that were extracted."""
#     train_or_test, classname, filename_no_ext, _ = video_parts
#     generated_files = glob.glob(os.path.join(train_or_test, classname,
#                                 filename_no_ext + '*.jpg'))

# split video into frames
def split_into_frames(data, chunkk):
    split_frames = []
    for video in data:
        split_frames = split_frames + [video[i:i + chunkk] for i in range(0, len(video), chunkk)]
    split_frames = [item for item in split_frames if len(item) == chunkk]
    return split_frames


# frames_splitt = split_into_frames(data, 3)
# print(len(frames_splitt))
# sys.exit()
# frame splitting complete
# ----------------------------------------------------------------------------------------------------------------------

# df_labels = [1] * len(data)

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
def extract_face(dataset_name='Dataset_seconds_videos', new_dataset_name='Dataset_Face_Extracted', new_dataset_name0='Dataset_Face_Extracted_picS', bbox_bias=30, bbox_size=299,
                 frames=15, extract_as_imgs = False):
    output_folder = new_dataset_name
    data_file = []
    # folders = [dataset_name + os.path.sep + 'Real' + os.path.sep, dataset_name + os.path.sep + 'Deepfake' + os.path.sep]
    folders = [dataset_name + os.path.sep + 'Deepfake' + os.path.sep]

    if not os.path.exists(new_dataset_name):
        os.mkdir(new_dataset_name)
    if not os.path.exists(new_dataset_name + os.path.sep + 'Deepfake'):
        os.mkdir(new_dataset_name + os.path.sep + 'Deepfake')
    if not os.path.exists(new_dataset_name + os.path.sep + 'Real'):
        os.mkdir(new_dataset_name + os.path.sep + 'Real')

    illus = 0

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
extract_face()
sys.exit()



# sys.exit()
# get_video_parts(os.getcwd()[:-5]+'\\Dataset\\Deepfake\\0.mp4')
# print(os.getcwd())

def extract_frames_to_images(dataset_name='Dataset_Face_Extracted_noise', new_dataset_name='Dataset_Face_Extracted_noise_img'):
    data_file = []
    folders = [dataset_name + os.path.sep + 'Real' + os.path.sep, dataset_name + os.path.sep + 'Deepfake' + os.path.sep]

    # some folder creation
    # print(os.path.exists('dataset_pre'))
    if not os.path.exists(new_dataset_name):
        os.mkdir(new_dataset_name)
    if not os.path.exists(new_dataset_name + os.path.sep + 'Deepfake'):
        os.mkdir(new_dataset_name + os.path.sep + 'DeepFake')
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

                # call([
                #     "C:\\Users\\Amine_Boukrout\\PycharmProjects\\ffmpeg-20191122-89aa134-win64-static\\bin\\ffmpeg",
                #     "-i", src, dest_folder_files])

                call([
                    "/local/java/ffmpeg/ffmpeg", "-i", src, dest_folder_files
                ])


            no_of_frames = get_nb_frames_for_video(new_dataset_name, video_parts)
            data_file.append([class_, file_name, no_of_frames])
            print('Video file {} generated {} files of frames'.format(file_name, no_of_frames))

    with open('data_info.csv', 'w') as out:
        writer = csv.writer(out)
        writer.writerows(data_file)

    print("Extracted and wrote %d video files." % (len(data_file)))
# extract_frames_to_images()
# sys.exit()


import os
import random
from shutil import copyfile, move, copytree

def img_train_test_split_fileS(img_source_dir, train_size):
    """
    Randomly splits images over a train and validation folder, while NOT preserving the folder structure

    Parameters
    ----------
    img_source_dir : string
        Path to the folder with the images to be split. Can be absolute or relative path

    train_size : float
        Proportion of the original images that need to be copied in the subdirectory in the train folder
    """
    if not (isinstance(img_source_dir, str)):
        raise AttributeError('img_source_dir must be a string')

    if not os.path.exists(img_source_dir):
        raise OSError('img_source_dir does not exist')

    if not (isinstance(train_size, float)):
        raise AttributeError('train_size must be a float')

    # Set up empty folder structure if not exists
    if not os.path.exists('data'):
        os.makedirs('data')
    else:
        if not os.path.exists('data/train'):
            os.makedirs('data/train')
        if not os.path.exists('data/validation'):
            os.makedirs('data/validation')

    # Get the subdirectories in the main image folder
    subdirs = [subdir for subdir in os.listdir(img_source_dir) if os.path.isdir(os.path.join(img_source_dir, subdir))]

    for subdir in subdirs:
        subdir_fullpath = os.path.join(img_source_dir, subdir)
        if len(os.listdir(subdir_fullpath)) == 0:
            print(subdir_fullpath + ' is empty')
            print(subdir_fullpath + ' is empty')
            break

        train_subdir = os.path.join('data/train', subdir)
        validation_subdir = os.path.join('data/validation', subdir)

        # Create subdirectories in train and validation folders
        if not os.path.exists(train_subdir):
            os.makedirs(train_subdir)

        if not os.path.exists(validation_subdir):
            os.makedirs(validation_subdir)

        train_counter = 0
        validation_counter = 0

        # Randomly assign an image to train or validation folder
        img_folders = os.listdir(subdir_fullpath)
        # for img_folder in img_folders:
        #     if random.uniform(0, 1) <= train_size:
        #         for filename in os.listdir(os.path.join(subdir_fullpath, img_folder)):
        #             if filename.endswith(".jpg") or filename.endswith(".png"):
        #                 fileparts = filename.split('.')
        #
        #             copyfile(os.path.join(subdir_fullpath, img_folder, filename),
        #                     os.path.join(train_subdir, str(train_counter) + '.' + fileparts[1]))
        #             train_counter += 1
        #     else:
        #         for filename in os.listdir(os.path.join(subdir_fullpath, img_folder)):
        #             if filename.endswith(".jpg") or filename.endswith(".png"):
        #                 fileparts = filename.split('.')
        #
        #             copyfile(os.path.join(subdir_fullpath, img_folder, filename),
        #                      os.path.join(validation_subdir, str(validation_counter) + '.' + fileparts[1]))
        #             validation_counter += 1
        #
        #     print('Copied ' + str(train_counter) + ' images to data/train/' + subdir)
        #     print('Copied ' + str(validation_counter) + ' images to data/validation/' + subdir)

        for img_folder in img_folders:
            random_uniform = random.uniform(0, 1)
            for filename in os.listdir(os.path.join(subdir_fullpath,img_folder)):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    fileparts = filename.split('.')

                    if random_uniform <= train_size:
                        copyfile(os.path.join(subdir_fullpath, img_folder, filename),
                                 os.path.join(train_subdir, str(train_counter) + '.' + fileparts[1]))
                        train_counter += 1
                    else:
                        copyfile(os.path.join(subdir_fullpath, img_folder, filename),
                                 os.path.join(validation_subdir, str(validation_counter) + '.' + fileparts[1]))
                        validation_counter += 1

            print('Copied ' + str(train_counter) + ' images to data/train/' + subdir)
            print('Copied ' + str(validation_counter) + ' images to data/validation/' + subdir)

# print('here')
# img_train_test_split_fileS('dataset_pre_images_',.8)
# sys.exit()

def img_train_test_split_folderS(img_source_dir, train_size):
    """
    Randomly splits images over a train and validation folder, while preserving the folder structure

    Parameters
    ----------
    img_source_dir : string
        Path to the folder with the images to be split. Can be absolute or relative path

    train_size : float
        Proportion of the original images that need to be copied in the subdirectory in the train folder
    """
    if not (isinstance(img_source_dir, str)):
        raise AttributeError('img_source_dir must be a string')

    if not os.path.exists(img_source_dir):
        raise OSError('img_source_dir does not exist')

    if not (isinstance(train_size, float)):
        raise AttributeError('train_size must be a float')

    # Set up empty folder structure if not exists
    if not os.path.exists('data'):
        os.makedirs('data')
    else:
        if not os.path.exists('data/train'):
            os.makedirs('data/train')
        if not os.path.exists('data/validation'):
            os.makedirs('data/validation')

    # Get the subdirectories in the main image folder
    subdirs = [subdir for subdir in os.listdir(img_source_dir) if os.path.isdir(os.path.join(img_source_dir, subdir))]

    for subdir in subdirs:
        subdir_fullpath = os.path.join(img_source_dir, subdir)
        if len(os.listdir(subdir_fullpath)) == 0:
            print(subdir_fullpath + ' is empty')
            break

        train_subdir = os.path.join('data/train', subdir)
        validation_subdir = os.path.join('data/validation', subdir)

        # Create subdirectories in train and validation folders
        if not os.path.exists(train_subdir):
            os.makedirs(train_subdir)

        if not os.path.exists(validation_subdir):
            os.makedirs(validation_subdir)

        train_counter = 0
        validation_counter = 0

        # Randomly assign an image to train or validation folder
        img_folders = os.listdir(subdir_fullpath)

        for img_folder in img_folders:
                if random.uniform(0, 1) <= train_size:
                    copytree(os.path.join(subdir_fullpath, img_folder),
                             os.path.join(train_subdir, str(train_counter)))
                    train_counter += 1
                else:
                    copytree(os.path.join(subdir_fullpath, img_folder),
                             os.path.join(validation_subdir, str(validation_counter)))
                    validation_counter += 1

        print('Copied ' + str(train_counter) + ' folders to data/train/' + subdir)
        print('Copied ' + str(validation_counter) + ' folders to data/validation/' + subdir)


# img_train_test_split_folderS('dataset_pre_images',.7)

# adding noise to images
def noisy(noise_typ, image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
          vals = len(np.unique(image))
          vals = 2 ** np.ceil(np.log2(vals))
          noisy = np.random.poisson(image * vals) / float(vals)
          return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)
        noisy = image + image * gauss
        return noisy

def noisy_data(data_folder='dataset_pre_images'):
    classes = ['DeepFake', 'Real']
    if not os.path.exists(os.path.join(data_folder+'_')):
        os.mkdir(os.path.join(data_folder + '_'))
    if not os.path.exists(os.path.join(data_folder+'_',classes[0])):
        os.mkdir(os.path.join(data_folder + '_',classes[0]))

    for clas in classes:
        if not os.path.exists(os.path.join(data_folder + '_', clas)):
            os.mkdir(os.path.join(data_folder + '_', clas))

        vid_img_flds = os.listdir(os.path.join(data_folder, clas))

        for vid_img_fld in vid_img_flds:
            if not os.path.exists(os.path.join(data_folder + '_', clas, vid_img_fld)):
                os.mkdir(os.path.join(data_folder + '_', clas, vid_img_fld))

            frames_file_names = os.listdir(os.path.join(data_folder, clas, vid_img_fld))
            for frame_file_name in frames_file_names:
                print(os.path.join(data_folder, clas, vid_img_fld, frame_file_name))
                image = cv2.imread(os.path.join(data_folder, clas, vid_img_fld, frame_file_name))
                image = noisy('gauss', image)
                cv2.imwrite(os.path.join(data_folder + '_', clas, vid_img_fld, frame_file_name), image)
# noisy_data()

import pandas as pd
def create_csv(data_folder='Dataset_Face_Extracted_clean_img'):
    data = []
    classes = ['Deepfake', 'Real']
    for clas in classes:
        class_dirr = os.path.join(data_folder, clas)
        vid_img_flds = os.listdir(class_dirr)

        for vid_img_fld in vid_img_flds:
            vid_img_fld_dirs = os.listdir(os.path.join(data_folder, clas, vid_img_fld))

            for vid_img_fld_dir_file in vid_img_fld_dirs:
                the_img = os.path.join(data_folder, clas, vid_img_fld, vid_img_fld_dir_file)
                data.append([vid_img_fld, the_img, clas])

    # pd.DataFrame.to_csv((''))
    df = pd.DataFrame(data, columns = ['video_file_name', 'file', 'label'])
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv('data.csv',sep=',')
# create_csv()
# sys.exit()

def create_csv_vids(data_folder='Dataset_Face_Extracted'):
    data = []
    classes = ['Deepfake', 'Real']
    for clas in classes:
        vid_class = os.path.join(data_folder, clas)
        vids_in_class = os.listdir(vid_class)
        for vid in vids_in_class:
            if str(vid).endswith('.mp4'):
                video = os.path.join(vid_class, vid)
                video_name = vid[:-4]
                label_one_hot = '0' if clas is 'Deepfake' else '1'
                data.append([video_name, video, clas, label_one_hot])

    df = pd.DataFrame(data, columns = ['video_file_name', 'file', 'label', 'label_one_hot'])
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv('df_videos.csv', sep=',')
# print('Creating CSV...')
# create_csv_vids()
# sys.exit()

def create_train_test_df(df='df_videos.csv'):
    df = pd.read_csv(df)
    videos_name = list(set(df.video_file_name))
    random.shuffle(videos_name)
    random.shuffle(videos_name)
    series = pd.Series(videos_name, name='video_file_name')

    np.random.seed(42)
    msk = np.random.rand(len(series)) < 0.75
    # msk = len(series) < 0.75
    series_train = series[msk]
    series_test = series[~msk]
    # print (list(series_train))

    df_train_temp = df[df['video_file_name'].isin(list(set(series_train)))]
    df_test_temp = df[df['video_file_name'].isin(list(set(series_test)))]

    df_train = pd.DataFrame(columns=df_train_temp.columns)
    df_test = pd.DataFrame(columns=df_test_temp.columns)

    i = 0
    train_videos = list(set(df_train_temp.video_file_name))
    for idx in range(len(train_videos)):
        # row = df_train_temp.iloc[idx]
        # name = row['video_file_name']
        video_name = str(train_videos[idx])
        df_temp = pd.DataFrame(df[df['video_file_name'] == video_name])
        count = 0
        for ii, row in df_temp.iterrows():
            df_train.loc[i] = row.values.tolist()
            i += 1
            count += 1
            if count > 9: break

    i = 0
    test_videos = list(set(df_test_temp.video_file_name))
    for idx in range(len(test_videos)):
        video_name = str(test_videos[idx])
        df_temp = pd.DataFrame(df[df['video_file_name'] == video_name])
        count = 0
        for ii, row in df_temp.iterrows():
            df_test.loc[i] = row.values.tolist()
            i += 1
            count += 1
            if count > 9: break

    print ('df_train has {} records'.format(len(df_train)))
    print ('df_test has {} records'.format(len(df_test)))

    df_test.to_csv('dataset_split/test_df.csv', sep=',')
    df_train.to_csv('dataset_split/train_df.csv', sep=',')
    return df_train, df_test
# create_train_test_df()
# sys.exit()



import random
def add_noise(image,k=1,std=0):
    print (image.shape)
    return cv2.GaussianBlur(image, (k,k), std)

def extract_video_to_no_of_frames(dataset_IMGS='dataset_pre_images', dataset_IMGS_new = 'Dataset_Face_Extracted_clean', split_no_frmes_split=20, fps=1, to_add_noise=False):
    classes = ['Deepfake', 'Real']
    if not os.path.isdir(dataset_IMGS_new):
        os.mkdir(dataset_IMGS_new)
    if not os.path.isdir(os.path.join(dataset_IMGS_new, classes[0])):
        os.mkdir(os.path.join(dataset_IMGS_new, classes[0]))
    if not os.path.isdir(os.path.join(dataset_IMGS_new, classes[1])):
        os.mkdir(os.path.join(dataset_IMGS_new, classes[1]))

    for classe in classes:
        the_videos = os.listdir(dataset_IMGS + os.path.sep + classe + os.path.sep)

        for img in the_videos:
            path_in = os.path.join(dataset_IMGS, classe, img)
            path_out = os.path.join(dataset_IMGS_new, classe, img)

            print ('\n',path_in,'\t',path_out,'\n\n')

            if to_add_noise:
                kernel_value = random.randrange(0, 4)
                kernel_values = [1, 3, 5, 7]
                k = kernel_values[kernel_value]
                std = random.randrange(0, 6)

            video_input = cv2.VideoCapture(path_in)
            video_input_frames = []
            while True:
                ret, frame = video_input.read()
                if to_add_noise and ret: frame = add_noise(frame,k,std)
                # print(type(frame)); return # sys.exit()
                if not ret: break
                video_input_frames.append(frame)
            video_input.release()

            if len(video_input_frames)>9:
                out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*'DIVX'), fps, (100,100))
                for k in range(len(video_input_frames)):
                    out.write(video_input_frames[k])
                out.release()

# extract_video_to_no_of_frames(dataset_IMGS='Dataset_Face_Extracted',dataset_IMGS_new='Dataset_Face_Extracted_noise',fps=0.5, to_add_noise=True)

def split(csv_file):
    test_or_train = ''
    if 'train' in csv_file: test_or_train = 'train'
    if 'test' in csv_file: test_or_train = 'test'
    print ('Working with {} set'.format(test_or_train))
    # time.sleep(5)

    dst_real = os.path.join('dataset_split', test_or_train, 'Real')
    dst_deepfake = os.path.join('dataset_split', test_or_train, 'Deepfake')

    src_real = os.path.join('Dataset_Face_Extracted', 'Real')
    src_deepfake = os.path.join('Dataset_Face_Extracted', 'DeepFake')

    # if not os.path.isdir(dst_real):
    #     os.mkdir(dst_real)
    # if not os.path.isdir(dst_deepfake):
    #     os.mkdir(dst_deepfake)

    df = pd.read_csv(csv_file, converters={'video_file_name': lambda x: str(x)})
    video_names = list(set(df.video_file_name))

    import shutil
    for video_name in video_names:
        video_label = df[df.video_file_name == video_name].values
        video_base_dst = dst_deepfake if video_label[0][4] == 'Deepfake' else dst_real
        video_base_src = src_deepfake if video_label[0][4] == 'Deepfake' else src_real
        dst_video = str(os.path.join(video_base_dst, str(video_name) + '.mp4'))
        src_video = video_label[0][3]# str(os.path.join(video_base_src, str(video_name) + '.mp4')).replace('DeepFake','Deepfake')

        print(df[df.video_file_name == video_name].file)
        print(video_name)

        shutil.copy(src_video, dst_video)
        print('Copied {} --> {}'.format(src_video, dst_video))

# split('dataset_split/train_df.csv')
# split('dataset_split/test_df.csv')
# sys.exit()
'''df_train = pd.read_csv('train_df.csv', converters={'video_file_name': lambda x: str(x)})
real_train_folders = set(df_train[df_train.columns[2]][df_train.label == 'Real'])
fake_train_folders = set(df_train[df_train.columns[2]][df_train.label == 'DeepFake'])

df_test = pd.read_csv('test_df.csv', converters={'video_file_name': lambda x: str(x)})
real_test_folders = set(df_test[df_test.columns[2]][df_test.label == 'Real'])
fake_test_folders = set(df_test[df_test.columns[2]][df_test.label == 'DeepFake'])

train_files_dir_real = 'dataset_split_clean/train/Real/'
train_files_dir_fake = 'dataset_split_clean/train/Deepfake/'

test_files_dir_real = 'dataset_split_clean/test/Real/'
test_files_dir_fake = 'dataset_split_clean/test/Deepfake/'


train_files_fake = os.listdir(train_files_dir_fake)
train_files_real = os.listdir(train_files_dir_real)

test_files_fake = os.listdir(test_files_dir_fake)
test_files_real = os.listdir(test_files_dir_real)

counnt = 0
for folder in real_test_folders:
    if folder in test_files_real:
        print (folder)
        counnt+=1

print('\n',counnt)
print (len(test_files_real))'''

def rescale_list(input_list, size):
    assert (len(input_list)) >= size
    skip = len(input_list)//size

    output = [input_list[i] for i in range(0, len(input_list), skip)]
    return output[:size]

def build_image_sequence(frames): # l 182
    return [process_image_arr(x, (299, 299, 3)) for x in frames]

# dff = pd.read_csv('dataLSTM/df_dataLSTM.csv', converters={'video_file_name': lambda x: str(x)})

#this will work only for raw images, not numpy arrays
def get_frames_of_sample(file_name, dataframe):
    frames = []
    df = dataframe[dataframe.video_file_name == file_name]

    for img_pth in df.file:
        img = cv2.imread(img_pth)
        img = cv2.resize(img, (299,299))
        frames.append(img)
    return frames
# get_frames_of_sample('23', dff)

import subprocess
def split_video_second(seconds, df):
    seconds = int(seconds)
    ffmpeg = "/local/java/ffmpeg/ffmpeg"
    df = pd.DataFrame(df)
    files = list(df.file)
    for file in files:
        print(file)
        subprocess.call(["python3", "ffmpeg-split.py", "-f", file, "-s", str(seconds)])
        try: os.remove(file)
        except: print('no such file!')

# df = pd.read_csv('data.csv')
# df = df[df.label == 'Real']
# split_video_second(1, df)
# sys.exit()

def delete_no_fps(data_folder='Dataset_work'):
    classes = ['Deepfake', 'Real']
    for cls in classes:
        cls_dir = os.path.join(data_folder,cls)
        files = os.listdir(cls_dir)
        for file in files:
            cap = cv2.VideoCapture(os.path.join(cls_dir,file))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps>0:
                duration = frame_count/fps
                print(duration)
            else:
                os.remove(os.path.join(cls_dir,file))
# delete_no_fps(data_folder='Dataset_Face_Extracted')

'''data = []
classes = ['Deepfake', 'Real']
for clas in classes:
    dir = os.path.join('Dataset_Face_Extracted', clas)
    files = os.listdir(dir)
    for file in files:
        the_file = os.path.join(dir,file)
        data.append([file,the_file,clas])
data = pd.DataFrame(data,columns=['video_file_name','file','label'])
data.to_csv('data.csv',columns=['video_file_name','file','label'])
print('done')'''

def get_all_sequences_in_memory(dataframe,col_filename,npy_file=True):
    video_sequences = []
    labels = []
    images_files_prefix = list(set(dataframe[col_filename]))

    for img in images_files_prefix:
        label = dataframe[dataframe[col_filename] == img].label.iloc[0]
        if label == 'Real': labels.append(1)
        else: labels.append(0)
        if npy_file:
            frames = []
            npy_file_ldd = np.load(dataframe[dataframe[col_filename] == img].file_pth)
            print(dataframe[dataframe[col_filename] == img].file_pth)
            sys.exit()
        else: frames = get_frames_of_sample(img,dataframe,col_filename)
        video_sequences.append(frames)
        del frames

    return np.array(video_sequences), np.array(labels)
# vids, .labs = get_all_sequences_in_memory(dff,'file_pth')
# print(len(vids)==len(labs))

def get_train_test_split(dataframe):
    videos = []
    for file in list(dataframe.video_file_name):
        frames = get_frames_of_sample(file,dataframe)
        videos.append(frames)
    return videos

def get_extracted_sequence(sample):
    """Get the saved extracted features."""
    path = sample[2]
    if os.path.isfile(path):
        return np.load(path)
    else:
        return None

import threading
class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)

def threadsafe_generator(func):
    """Decorator"""
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen

from tensorflow.keras.utils import to_categorical
from keras_video import VideoFrameGenerator
# 3 videos in 30 frames -->
@threadsafe_generator
def frame_generator(dataframe, train_test, data_type, bs=5):
    # images_files_prefix = list(set(dataframe[col_filename]))
    # df_train = pd.DataFrame(dataframe[dataframe['split'] == 'train'])
    # df_test = pd.DataFrame(dataframe[dataframe['split'] == 'test'])

    if train_test is 'train':
        df = pd.read_csv('train_df.csv', converters={'video_file_name': lambda x: str(x)})
    elif train_test is 'test':
        df = pd.read_csv('test_df.csv', converters={'video_file_name': lambda x: str(x)})
    else:
        raise ValueError('The train_test parameter can only be \'train\' or \'test\'.')

    train, test = [], []
    for row in dataframe.iterrows():
        set = str(row[1]['split']).strip()
        video_file_name_pre = row[1]['video_file_name']
        label, label_one_hot = row[1]['label'], row[1]['label_one_hot']
        if data_type is 'images':
            frames = get_frames_of_sample(video_file_name_pre, df)
        elif data_type is 'data':
            frames = np.load(row[1]['file'])
        else:
            raise ValueError('data_type parameter can only be \'data\' or \'images\'.')

        if set == 'train':
            train.append((frames, label_one_hot, row[1]['file']))
        else:
            test.append((frames, label_one_hot, row[1]['file']))
    data = train if train_test is 'train' else test

    while 1:
        x,y = [],[]

        for _ in range(bs):
            sequence = None
            sample = random.choice(data)

            if data_type is 'images':
                frames = rescale_list(sample[0],10)
                sequence = build_image_sequence(frames)
            else:
                sequence = sample[0]
                if sequence is None:
                    raise ValueError('No sequence loaded!')
            x.append(sequence)
            if sample[1] == 0: labell = [1,0]
            else: labell = [0,1]
            y.append(labell)
        yield np.array(x), np.array(y)

def remove_videos_with_no_frames(csv='df_videos_faces_extracted.csv'):
    df = pd.read_csv(csv)
    for video_path in df.file_path:
        if os.path.isfile(video_path):
            input_video = vid_init(video_path)
            frame_count = 0
            while True:
                ret, frame = input_video.read()
                if not ret or frame_count>2:
                    break
                frame_count += 1
                rgb_frame = frame  # cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
            # if frame_count==0:
            #     if os.path.isfile(video_path):
            #         os.remove(video_path)
            print(frame_count)
# remove_videos_with_no_frames(

import glob
def img_to_vids(data_folder='Dataset_Face_Extracted'):
    new_folder = data_folder + '_videos' + os.path.sep
    if not os.path.isdir(new_folder): os.mkdir(new_folder)
    if not os.path.isdir(new_folder+'Deepfake'): os.mkdir(new_folder+'Deepfake')
    if not os.path.isdir(new_folder+'Real'): os.mkdir(new_folder+'Real')
    print('hhhhhhhhhhhhh')
    count = 0

    classes = ['Deepfake', 'Real']
    for clas in classes:
        class_folder = os.path.join(data_folder,clas)
        vid_names = os.listdir(class_folder)
        for vid_name in vid_names:
            vid_imgs_folder = os.path.join(class_folder,vid_name)
            img_array = []
            size = (224, 224)
            if os.path.isdir(vid_imgs_folder):
                vid_imgs = os.listdir(vid_imgs_folder)
                for vid_img in vid_imgs:
                    # print(os.path.join(vid_imgs_folder,vid_img))
                    img = cv2.imread(os.path.join(vid_imgs_folder,vid_img))
                    height, width, layers = img.shape
                    size = (width,height)
                    img_array.append(img)
                    # print(size,'\n',img_array[0].shape)
                    # sys.exit()
                print(os.path.join(new_folder,clas,vid_name))
                out = cv2.VideoWriter(os.path.join(new_folder,clas,vid_name+'.mp4'),cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
                # sys.exit()
                for i in range(len(img_array)):
                    out.write(img_array[i])
                    print('i=',i)
                out.release()
# img_to_vids()

import shutil
def move_files_to_split_folder(data_folder='Dataset_Face_Extracted', moveto_folder='dataLSTM/data', split='test'):
    classes = ['Real', 'Deepfake']

    if split is 'train':
        df = pd.read_csv('train_df.csv', converters={'video_file_name': lambda x: str(x)})
    else:
        df = pd.read_csv('test_df.csv', converters={'video_file_name': lambda x: str(x)})
    video_files = set(list(df.video_file_name))

    for clas in classes:
        class_base = os.path.join(data_folder, clas)
        class_vids = os.listdir(class_base)
        for vid in class_vids:
            if str(vid).split('.')[0] in video_files:
                vid_dir = os.path.join(class_base,vid)
                move_to_dir = os.path.join(moveto_folder, split, clas, vid)
                shutil.copy(vid_dir, move_to_dir)
                print('Copied {} --> {}'.format(vid_dir,move_to_dir))
# move_files_to_split_folder()
# traindf
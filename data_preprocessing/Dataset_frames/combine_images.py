import os
import sys
import glob
import subprocess
import cv2

def video_duration(video):
    # print(os.path.isfile(video))
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(frame_count/fps); sys.exit()
    duration = frame_count / fps

    print('fps = ' + str(fps))
    print('number of frames = ' + str(frame_count))
    print('duration (S) = ' + str(duration))
    minutes = int(duration / 60)
    seconds = duration % 60
    print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))

    cap.release()

def combine_images(main_folder = 'Dataset_final_images'):
    import pathlib
    original_curdir = pathlib.Path(__file__).parent.absolute()

    folders = [os.path.join(main_folder,'Deepfake'), os.path.join(main_folder,'Real')]
    for folder in folders:
        if not os.path.isdir(folder+'_'): os.mkdir(folder+'_')
        video_folders = os.listdir(folder)
        # print(len(video_folders))
        for video_folder in video_folders:
            video_file = os.path.join(original_curdir,folder+'_',video_folder+'.mp4')
            input = os.path.join(folder, video_folder, '-%04d.jpg')
            # print(video_folder)

            # print(os.path.isdir(os.path.join(folder, video_folder)))
            # print(folder,'\n',video_folder)
            # print('data_preprocessing/Dataset_frames/Dataset_final_images/Deepfake/videoplayback-9')
            # print(os.path.isdir('data_preprocessing/Dataset_frames/Dataset_final_images/Deepfake/videoplayback-9'))
            os.chdir(os.path.join(folder, video_folder))

            print(os.listdir(os.curdir))
            # print(video_folder[:-3]+'-%1d.jpg')
            print(video_folder)
            print(video_file)
            video_file = str(video_file).replace('(','-').replace(')','-')
            # sys.exit()
            # command = "/local/java/ffmpeg/ffmpeg -r 20 -i '{}' -vcodec mpeg4 -y {}".format(video_folder+'-%04d.jpg', video_file)
            command = "/local/java/ffmpeg/ffmpeg -framerate 20 -pattern_type glob -i '*.jpg' -vcodec mpeg4 -y {}".format(video_file)
            subprocess.call(command, shell=True)
            video_duration(video_file)
            os.chdir(original_curdir)
            # print('jjjjjjjjjjjjjjjj')
            # print(os.listdir(os.curdir))
            # print(os.path.isfile(video_file))
            # sys.exit()
combine_images()
sys.exit()

def ccombine_images():
    # size = (720, 1280)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    fps = 1
    # sys.exit()

    folders = ['Deepfake', 'Real']
    for folder in folders:
        video_folders = os.listdir(folder)

        for video_folder in video_folders:
            video_frames = glob.glob(os.path.join(folder,video_folder,'*.jpg'))
            video_frames = sorted(video_frames)
            output_file = os.path.join(folder+'_',video_folder+'.avi')

            frame_array = []
            size = 0

            for video_frame in video_frames:
                img = cv2.imread(video_frame)
                height, width, layers = img.shape
                size = (height, width)
                print('hhhhh')
                frame_array.append(img)

            vout = cv2.VideoWriter(output_file, fourcc, fps, size)
            for i in range(len(frame_array)):
                vout.write(frame_array[i])

            vout.release()
            sys.exit()
# ccombine_images()

def copy_images(main_folder = 'face_images'):
    import pathlib
    import shutil
    original_curdir = pathlib.Path(__file__).parent.absolute()

    folders = [os.path.join(main_folder, 'Deepfake'), os.path.join(main_folder, 'Real')]
    output_dataset = 'Dataset_final_images'

    for folder in folders:
        main_folder_, clas = folder.split(os.path.sep)
        video_image_folders = os.listdir(folder)
        for video_image_folder in video_image_folders:
            frames_images_dirs = sorted(glob.glob(os.path.join(folder,video_image_folder,'*.jpg')))
            video_written_counter = 0

            frames = []
            for i in range(len(frames_images_dirs)):
                # jpg = frames_images_dirs[i].split(os.path.sep)
                # print(jpg)
                # return
                frames.append(frames_images_dirs[i])
                # frame_counter = 0

                if len(frames) == 20 or i == len(frames_images_dirs)-1:
                    if len(frames) < 20:
                        while len(frames) != 20:
                            frames.append(frames[len(frames)-1])
                    print('video_written_counter: ' + str(video_written_counter))
                    new_folder = os.path.join(output_dataset, clas, video_image_folder+'-'+str(video_written_counter))
                    if not os.path.isdir(new_folder): os.mkdir(new_folder)

                    frame_counter = 0
                    for frame in frames:
                        # print(new_folder)
                        # print(video_written_counter)
                        new_file = video_image_folder + '-' + str(video_written_counter) + '-' + str(frame_counter) + '.jpg'
                        new_dir = os.path.join(new_folder, new_file)
                        # print(new_file)
                        # print(new_dir)
                        # sys.exit()
                        # print(frame)
                        # print(new_dir)
                        # sys.exit()

                        try:
                            shutil.copy(frame, new_dir)
                            print('Coppied {}'.format(new_dir))
                        except:
                            print('File {} not written'.format(new_dir))
                        frame_counter += 1
                    video_written_counter += 1
                    frames = []
copy_images()

def clean_mp4():
    folders = ['Deepfake_', 'Real_']
    for folder in folders:
        videos = glob.glob(os.path.join(folder,'*.mp4'))
        for video in videos:
            os.remove(video)
# clean_mp4()


# video_duration('Deepfake_'+os.path.sep+'BabyJoeRoganandBabyElonMuskSmokeWeed-Deepfake-15.avi')
# videos = glob.glob(os.path.join('Deepfake_','*.mp4'))
# for video in videos:
#     print(os.path.join(video))
#     video_duration(os.path.join(video))
#     print()

# video_duration('102-15.mp4')

see_equal_frames = False
if see_equal_frames:
    import numpy as np
    print(os.listdir(os.curdir))
    print(os.path.isfile('/dcs/pg19/u1964004/PycharmProjects/deepfakes/data_preprocessing/Dataset/Deepfake/0.mp4'))
    input_video = cv2.VideoCapture('/dcs/pg19/u1964004/PycharmProjects/deepfakes/data_preprocessing/Dataset/Deepfake/0.mp4')
    frame_number = 0
    frames = []
    while True:
        ret, frame = input_video.read()
        print(ret)
        # sys.exit()
        frames.append(frame)
        frame_number += 1
        if not ret: break
    print(frame_number)
    for f in range(len(frames)-1):
        print(f)
        print(np.array_equal(frames[f],frames[f+1]))
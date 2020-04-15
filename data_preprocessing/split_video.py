import os
import sys
import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip
import subprocess

def split_video(file_info='file_info.csv', clip_siize=1):
    if not os.path.isdir('Dataset_resampled'):
        os.mkdir('Dataset_resampled')
    if not os.path.isdir('Dataset_video_splited'):
        os.mkdir('Dataset_video_splited')
    if not os.path.isdir(os.path.join('Dataset_resampled','Deepfake')):
        os.mkdir(os.path.join('Dataset_resampled','Deepfake'))
    if not os.path.isdir(os.path.join('Dataset_resampled','Real')):
        os.mkdir(os.path.join('Dataset_resampled','Real'))
    if not os.path.isdir(os.path.join('Dataset_video_splited','Deepfake')):
        os.mkdir(os.path.join('Dataset_video_splited','Deepfake'))
    if not os.path.isdir(os.path.join('Dataset_video_splited','Real')):
        os.mkdir(os.path.join('Dataset_video_splited','Real'))

    file_info = pd.read_csv(file_info)
    dirs = list(file_info['viddir'])
    for filename in dirs:
        if filename.split('/')[2].endswith('.mp4'):
            video_filetype = '.mp4'
            command = '/local/java/ffmpeg/ffmpeg -i {} -r 20 -y {}'.format(filename, os.path.join('Dataset_resampled', filename.split('/')[1], filename.split('/')[2]))
            subprocess.call(command, shell=True)

            input_video_path = os.path.join('Dataset_resampled', filename.split('/')[1], filename.split('/')[2])
            # print(os.path.isfile(input_video_path))
            if os.path.isfile(input_video_path):
                with VideoFileClip(input_video_path) as video:
                    clip_count = int(video.duration/clip_siize)

                    for clip in range(clip_count):
                        output_video_path = os.path.join('Dataset_video_splited', filename.split('/')[1], str(filename.split('/')[2]).split('.')[0]+'-'+str(clip)+video_filetype)
                        start = clip
                        end = clip + clip_siize #clip_count
                        new = video.subclip(start, end)
                        new.write_videofile(output_video_path, audio=False, codec='libx264')
            else: print('Warning missing file {}'.format(filename))
        print('File split complete!')

split_video()

# import glob
# print(os.listdir(os.curdir))
# files = os.listdir(os.curdir)
# for file in glob.glob('*.mp4'):
#     os.remove(file)
#     print(file)
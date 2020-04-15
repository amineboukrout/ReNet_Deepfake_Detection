import os
import pandas as pd

def original_dataset():
    files_info = []
    main_dirs = [os.path.join('Dataset','Deepfake'), os.path.join('Dataset','Real')]
    for dir in main_dirs:
        files = os.listdir(dir)
        for file in files:
            file_dir = os.path.join(dir,file)
            files_info.append([dir.split(os.path.sep)[1],file,file_dir])

    df = pd.DataFrame(files_info,columns=['class','file','viddir'])
    df.to_csv('file_info.csv')

def splited_dataset():
    files_info = []
    main_dirs = [os.path.join('Dataset_video_splitted', 'Deepfake'), os.path.join('Dataset_video_splitted', 'Real')]
    for dir in main_dirs:
        files = os.listdir(dir)
        for file in files:
            file_dir = os.path.join(dir, file)
            files_info.append([dir.split(os.path.sep)[1], file, file_dir])

    df = pd.DataFrame(files_info, columns=['class', 'file', 'viddir'])
    df.to_csv('file_info_splitted.csv')
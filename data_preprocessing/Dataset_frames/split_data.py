import pandas as pd
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split

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
# create_csv_vids('Dataset_final')

def get_traintest_dfs(data_csv):
    df = pd.read_csv(data_csv)
    df.sample(frac=1)
    fracs = np.array([0.8, 0.2])
    train_df, test_df = np.array_split(
        df,(fracs[:-1].cumsum() * len(df)).astype(int)
    )

    train_df = pd.DataFrame(train_df)
    train_df.to_csv('train_df.csv')
    test_df = pd.DataFrame(test_df)
    test_df.to_csv('test_df.csv')
    return train_df, test_df
# get_traintest_dfs('df_videos.csv')

def get_traintest_dfs_1(data_csv):
    df = pd.read_csv(data_csv)
    df.sample(frac=1)
    y = df.pop('label').to_frame()
    df.pop('label_one_hot'), df.pop('video_file_name'), df.pop('Unnamed: 0')
    x_train, x_test, y_train, y_test = train_test_split(df['file'],y, stratify=y, test_size=0.2)
    return x_train, x_test, y_train, y_test
# get_traintest_dfs_1('df_videos.csv')

def move_to_split_folders(csv_file):
    df = pd.read_csv(csv_file)
    mew_data_folder = 'data'

    if not os.path.isdir('data/'): os.mkdir('data/')

    if not os.path.isdir(os.path.join('data','train')):
        os.mkdir(os.path.join('data','train'))
    if not os.path.isdir(os.path.join('data','test')):
        os.mkdir(os.path.join('data','test'))

    if not os.path.isdir(os.path.join(mew_data_folder,'train','Deepfake')):
        os.mkdir(os.path.join(mew_data_folder, 'train', 'Deepfake'))
    if not os.path.isdir(os.path.join(mew_data_folder, 'train', 'Real')):
        os.mkdir(os.path.join(mew_data_folder, 'train', 'Real'))
    if not os.path.isdir(os.path.join(mew_data_folder, 'test', 'Deepfake')):
        os.mkdir(os.path.join(mew_data_folder, 'test', 'Deepfake'))
    if not os.path.isdir(os.path.join(mew_data_folder, 'test', 'Real')):
        os.mkdir(os.path.join(mew_data_folder, 'test', 'Real'))

    test_or_train = ''
    if 'train' in csv_file: test_or_train = 'train'
    if 'test' in csv_file: test_or_train = 'test'
    print ('Working with {} set'.format(test_or_train))

    import shutil
    for row in df.iterrows():
        row = row[1]
        org_pth = os.path.join(row.file)
        file_parts = str(row.file).split('/')
        org_pth_new = os.path.join(mew_data_folder, test_or_train, row.label, str(row.video_file_name) + '.mp4')

        shutil.copy(org_pth, org_pth_new)
        print('Copied {} --> {}'.format(org_pth, org_pth_new))
# move_to_split_folders('train_df.csv')
# move_to_split_folders('test_df.csv')

# if __name__ == '___main__':
#     get_traintest_dfs_0('df_videos.csv')
#     move_to_split_folders('train_df.csv')
#     move_to_split_folders('test_df.csv')

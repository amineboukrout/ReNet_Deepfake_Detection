import sys
from keras.models import Model, load_model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.layers import LSTM, Input, Conv2D, BatchNormalization, MaxPool2D, GlobalMaxPool2D, TimeDistributed, Dense, Dropout
import numpy as np
import pandas as pd
import glob,os
from tensorflow.keras.applications import InceptionResNetV2
from keras.applications.inception_v3 import preprocess_input, InceptionV3
from tqdm import tqdm
from pytvid import get_frames_of_sample, rescale_list
import cv2

batch_size = 10

def load_data():
    datagen = ImageDataGenerator(rescale=1./255)
    train_generator = datagen.flow_from_dataframe(
        dataframe = pd.read_csv('train_df.csv'),
        x_col = 'file',
        y_col = 'label',
        target_size=(128,128),
        batch_size=batch_size,
        class_mode='categorical',  # this means our generator will only yield batches of data, no labels
        shuffle=False,
        classes= ['Real','DeepFake']
    )

    test_generator = datagen.flow_from_dataframe(
        dataframe = pd.read_csv('test_df.csv'),
        x_col = 'file',
        y_col = 'label',
        target_size= (128,128),
        batch_size=batch_size,
        class_mode='categorical',  # this means our generator will only yield batches of data, no labels
        shuffle=False,
        classes= ['Real','DeepFake']
    )
    return train_generator, test_generator
# tg, vg = load_data()
# print (len(vg[0]))
# print (vg[1])
# sys.exit()

def base_model(input_shape = (128, 128, 3)):
    googleNet_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    googleNet_model.trainable = True
    print ('Model loaded...\n',googleNet_model.summary())
    return googleNet_model


class Extractor():
    def __init__(self, image_shape=(299, 299, 3),weights=None):
        """Either load pretrained from imagenet, or load our saved
        weights from our own training."""
        self.weights = weights  # so we can check elsewhere which model

        input_tensor = Input(image_shape)
        # Get model with pretrained weights.
        base_model = InceptionV3(
            input_tensor=input_tensor,
            weights='imagenet',
            include_top=True
        )

        # We'll extract features at the final pool layer.

        self.model = Model(
            inputs=base_model.input,
            outputs=base_model.get_layer('avg_pool').output
        )

    def extract(self, image_path):
        img = image.load_img(image_path)
        return self.extract_image(img)

    def extract_image(self, img):
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Get the prediction.
        features = self.model.predict(x)

        if self.weights is None:
            # For imagenet/default network:
            features = features[0]
        else:
            # For loaded network:
            features = features[0]
        return features


def remove_features_from_dir(data_csv,col_file):
    train_or_test = 'train'
    # if 'train' in data_csv: train_or_test = 'train'
    # else: train_or_test = 'test'

    df = pd.read_csv(data_csv, converters={col_file: lambda x: str(x)})
    the_videos = list(set(df[col_file]))

    for video_file in the_videos:
        label = df[df[col_file] == video_file].label.iloc[0]
        folder_path = os.path.join('dataLSTM', 'data', train_or_test, label.title(), video_file)
        npy_path = folder_path + '.npy'
        if os.path.isfile(npy_path):
            os.remove(npy_path)
            print('Deleted {}.'.format(npy_path))



# this function CANNOT load npy features from file
def extract_features(data_csv, seq_length=10, class_limit=2, image_shape = (128, 128, 3)):
    remove_features_from_dir(data_csv, 'video_file_name')

    train_or_test = None
    if 'train' in data_csv: train_or_test = 'train'
    else: train_or_test = 'test'

    model = Extractor()
    df = pd.read_csv(data_csv, converters={'video_file_name': lambda x: str(x)})
    pbar = tqdm(total = len(df))
    the_videos = set(df['video_file_name'])

    for video_file in the_videos:
        label = df[df['video_file_name'] == video_file].label.iloc[0]
        # folder_path = os.path.join('dataset_split_clean', train_or_test, label.title(), video_file)
        folder_path_npy = os.path.join('dataLSTM', 'data', train_or_test, label.title(), video_file)
        #    npy_path = os.path.join(folder_path,video_file+'.npy')
        # video_frames = os.listdir(folder_path)
        # print (os.path.isfile(folder_path_npy+'.npy'))
        # print (folder_path_npy+'.npy')
        # return

        if os.path.isfile(folder_path_npy+'.npy'):
            pbar.update(10)
            continue

        # print(video_file)
        frames = get_frames_of_sample(video_file,df,'video_file_name')

        # print(len(frames))
        frames = rescale_list(frames, seq_length)
        # print(len(frames))
        # sys.exit()

        sequence = []

        for image in frames:
            features = image #model.extract_image(image)
            sequence.append(features)

        np.save(folder_path_npy, sequence)
        pbar.update(10)
    pbar.close()

# extract_features('train_df.csv')
# extract_features('test_df.csv')

def features_csv(data_folder='dataLSTM/data'):
    train_test = os.listdir(data_folder)
    data = []
    for tt in train_test:
        tt_path = os.path.join(data_folder,tt)
        tt_classes = os.listdir(tt_path)
        # print(tt_classes)
        # print(tt_path)
        for tt_class in tt_classes:
            tt_files_path = os.path.join(tt_path,tt_class)
            tt_files = os.listdir(tt_files_path)

            for tt_file in tt_files:
                npy_file = os.path.join(tt_files_path,tt_file)
                file_prefix = str(npy_file).split(os.path.sep)[-1].split('.')[0]
                label_one_hot = 0 if tt_class == 'Deepfake' else 1
                data.append([file_prefix,npy_file,tt,tt_class,label_one_hot])
    df = pd.DataFrame(data, columns=['video_file_name','file','split','label','label_one_hot'])
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv('dataLSTM/df_dataLSTM.csv', sep=',')
# features_csv()
# print('hhhhhhhhhhh')

def load_npy_features(dataframe, train_test='train'):
    df = pd.DataFrame(dataframe[dataframe['split'] == train_test])
    x, y = [], []
    df_work = pd.DataFrame(df[['file','label_one_hot']])
    for index, row in df_work.iterrows():
        img = np.load(str(row['file']))
        label = int(row['label_one_hot'])
        x.append(img)
        y.append(label)
        # print(row['file'],row['label_one_hot'])
    return x, y
dataframe = pd.read_csv('train_df.csv', converters={'video_file_name': lambda x: str(x)})
# x, y = load_npy_features(dataframe)
# print(x[0][0].shape)


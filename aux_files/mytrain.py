import sys

import os
import cv2
import json
from statistics import mode
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import load_model, model_from_json
import seaborn as sn
import dlib
from PIL import Image, ImageChops, ImageEnhance
import math

def load_dataset(df_name, batch_size=32, mode='train'):
    df = pd.read_csv(df_name)
    image_paths = list(df.file)

    x = []
    y = []

    for img in image_paths:
        x.append(img_to_array(load_img(img, target_size=(128,128,3))).flatten() / 255.0)
        if 'Real' in img: y.append(1)
        elif 'DeepFake' in img: y.append(0)

    y_val_org = y

    # normalization
    x = np.array(x)
    y = to_categorical(y, 2)
    x = x.reshape(-1, 128, 128, 3)

    if mode is 'train':
        trainX, valX, trainY, valY = train_test_split(x, y, test_size=0.2, random_state=42)

        return trainX, valX, trainY, valY
    else:
        return x, y_val_org

def load_video_sequence(img_name, file='test_df.csv'):
    df = pd.read_csv(file)
    # print (df.head(3))
    # print (set(df.video_file_name))
    # print (set(df.label))
    df = pd.DataFrame(df[df['video_file_name'] == img_name])
    # print (df)

    image_paths = list(df.file)

    x = []
    y = []

    for img in image_paths:
        x.append(img_to_array(load_img(img, target_size=(128,128,3))).flatten() / 255.0)
        if 'Real' in img: y.append(1)
        elif 'DeepFake' in img: y.append(0)

    y_val_org = y
    # print (len(y))

    # normalization
    x = np.array(x)
    y = to_categorical(y, 2)
    x = x.reshape(-1, 128, 128, 3)
    return x, y_val_org


# load_video_sequence(img_name='2')

# googleNet_model = InceptionResNetV2(weights='imagenet', include_top=True, input_shape=(128,128,3) )
# print (googleNet_model.summary())
# sys.exit()

def ggnModel(input_shape=(128,128,3)):
    googleNet_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    googleNet_model.trainable = True

    model = Sequential()
    model.add(googleNet_model)
    model.add(GlobalAveragePooling2D())
    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                  metrics=['accuracy'])
    model.add(Dense(2, activation='softmax'))
    print (model.summary())
    return model

def my_base_model(input_shape=(128, 128, 3)):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())

    # flatten
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False),
                  metrics=['accuracy'])
    print (model.summary())
    return model
# my_base_model()
# sys.exit()

def train(epochs=20, batch_size=32, patience=2):
    trainX, valX, trainY, valY = load_dataset('train_df.csv')
    early_stopping = EarlyStopping(monitor='val_loss',
                                   min_delta=0,
                                   patience=patience,
                                   verbose=0, mode='auto')

    model = ggnModel()
    model.fit(trainX, trainY, batch_size=batch_size, epochs=epochs, validation_data=(valX, valY), verbose=1, callbacks = [early_stopping])
    model.save_weights('myt_weights_softmax.h5')
    # model_json = model.to_json()
    # with open('model.json', 'w') as json_file:
    #     json_file.write(model_json)
# train(10)
# sys.exit()

#Output confusion matrix
def print_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print('True positive = ', cm[0][0])
    print('False positive = ', cm[0][1])
    print('False negative = ', cm[1][0])
    print('True negative = ', cm[1][1])
    print('\n')
    df_cm = pd.DataFrame(cm, range(2), range(2))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
    plt.ylabel('Actual label', size = 20)
    plt.xlabel('Predicted label', size = 20)
    plt.xticks(np.arange(2), ['Fake', 'Real'], size = 16)
    plt.yticks(np.arange(2), ['Fake', 'Real'], size = 16)
    plt.ylim([2, 0])
    plt.savefig('cm_softmax')

x, y = load_dataset('test_df.csv', mode='test')
# print (y,'\n',len(y))
# print (459%10)
# print (y[0])
# sys.exit()

# x, y = load_video_sequence('94')

modekl = ggnModel()
modekl.load_weights('my_weights_softmax.h5')
print_confusion_matrix(y, modekl.predict_classes(x))
# print (modekl.predict_classes(x))
# print (y)
sys.exit()


df = pd.read_csv('test_df.csv')

df = pd.DataFrame(df[['video_file_name', 'label']]).drop_duplicates(subset='video_file_name', keep='last')
ground_truth = []
for img in list(df['label']):
    if 'Real' in img:
        ground_truth.append(1)
    elif 'DeepFake' in img:
        ground_truth.append(0)
df['label1hot'] = ground_truth
# sys.exit()

# print (df.head())
videos = list(set(df.video_file_name))
ground_truth = list(df.label1hot)
pred_vals = []
for video in videos:
    x, y = load_video_sequence(str(video))
    fr_pred = modekl.predict_classes(x)
    try:
        pred_vals.append (mode(fr_pred))
    except:
        import random
        # print ('equal modes\n',fr_pred,'\n',int(random.sample(set('01'),1)[0]))
        pred_vals.append(int(random.sample(set('01'),1)[0]))
# print (len(ground_truth)==len(pred_vals))
# print_confusion_matrix(ground_truth, fr_pred)
# print (ground_truth,'\n',pred_vals)

correct = 0
for idx in range(len(ground_truth)):
    if ground_truth[idx] == pred_vals[idx]:
        correct += 1

print ('{} correct out of {}'.format(correct, len(ground_truth)))
print ('Model has accuracy {}'.format(correct/len(ground_truth)))
sys.exit()
for idx in range(len(vid_arr_labels)):
    y_arr = vid_arr_labels[idx]
    x_arr = vid_arr_frames[idx]
    predmn = modekl.predict_classes(x_arr)
    preds = []
    # for imgg in x_arr:
    #     print (imgg)
    #     predict = modekl.predict_classes(imgg)
    #     preds.append(predict)
    #     sys.exit()
    # print (len(preds))

    break
sys.exit()

def predict(filename, model):
    # model = ggnModel()
    # model.load_weights('my_weights.h5')
    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(filename)
    frameRate = cap.get(5)
    while cap.isOpened():
        frameID = cap.get(1)
        ret, frame = cap.read()
        if ret != True: break
        if frameID % ((int(frameRate)+1)*1) == 0:
            face_rects, scores, idx = detector.run(frame, 0)
            for i, d in enumerate(face_rects):
                x1 = d.left()
                y1 = d.top()
                x2 = d.right()
                y2 = d.bottom()
                crop_image = frame[y1:y2, x1:x2]
                data = np.array(img_to_array(cv2.resize(crop_image, (128, 128))))
                data = data.reshape(-1, 128, 128, 3)
                print(model.predict_classes(data))

# predict('Dataset/Real/000029301.mp4', modekl)
#
# df = pd.read_csv('test_df.csv')
# dfr = df[df['label'] == 'Real']
# pths = list(df.file)
# for pth in pths:
#     predict(pth, modekl)
#     print('------------------------------------------------------------------------------------------------------------')

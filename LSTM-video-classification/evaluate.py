from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data import DataSet
from extract_features import extract_features
import time
import os.path
import sys
import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth',-1)
import cv2
from clasify import classify, print_confusion_matrix
from extract_features import extract_features

# df = pd.read_csv('data/data_file.csv', converters={'file': lambda x: str(x)},header=None)
# df.columns = ['split','clas','file','no_of_frames']
# df = pd.DataFrame(df[df.split == 'test'])
# df = pd.DataFrame(df[df.clas == 'Deepfake'])
# df = pd.DataFrame(df[df.no_of_frames >= 10])
# print(df)
# sys.exit()

# video_preds = []
# for row in df.iterrows():
#     video = os.path.join('data',row[1].split,row[1].clas,str(row[1].file)+'.mp4')
#     actual_label = 0 if row[1].clas == 'Deepfake' else 1
#     video_preds.append([str(row[1].file)+'.mp4', int(actual_label), int(classify(video_file=video, seq_length=10, saved_model = 'cnn_lstm_VGGFace10_SPLIT0.h5'))])
#     print('\nlllllllllllllll\n', classify(video_file=video, seq_length=10, saved_model = 'cnn_lstm_VGGFace10_SPLIT0.h5'), '\nlllllllllllllll')
# print('len preds:',len(video_preds))

# dff = pd.DataFrame(video_preds,columns=['file','label','prediction'])
# dff.to_csv('results.csv')
# sys.exit()

IMGWIDTH = 224
sequences_dir = os.path.join('data', 'sequences')
if not os.path.exists(sequences_dir):
    os.mkdir(sequences_dir)

extract_features(10, 2, (224,224,3))
data = DataSet(
            seq_length=10,
            class_limit=2,
          image_shape=(IMGWIDTH,IMGWIDTH,3)
        )
data_type = 'features'
# train, test = data.split_train_test()
# print(test)
x, y = data.get_all_sequences_in_memory('test', data_type)

weights_file = 'cnn_lstm_VGGFace10_SPLIT4.h5'
the_model = ResearchModels(2,'lstm',10,features_length=2622)
the_model.model.load_weights(weights_file)
# results = the_model.model.evaluate(x_test_imgs,y_test, batch_size=32)
# print('test loss: {} \n test acc: {}'.format(results[0],results[1]))

# test_gen = data.frame_generator(32, 'test')
# for _ in range(5):
#     results = the_model.model.evaluate_generator(test_gen, steps=30)
#     print('test loss: {} \t test acc: {}'.format(results[0],results[1])

print(len(x))
preds = the_model.model.predict(x)
print(len(preds))
print(len(y))
print(preds)
print(y)

preds_onehot, y_onehot = [], []

assert len(x)==len(preds) and len(x)==len(y)
for i in range(len(x)):
    preds_onehot.append(np.argmax(preds[i]))
    y_onehot.append(np.argmax(y[i]))
    # print(preds[i],'\t',np.argmax(preds[i]))
    # print(y[i],'\t',np.argmax(y[i]))
    # print()
print(preds_onehot)
print(y_onehot)

print_confusion_matrix(y_onehot, preds_onehot)
from tensorflow.keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from tensorflow.keras.layers import LSTM, Input, Conv2D, BatchNormalization, MaxPool2D, GlobalMaxPool2D, TimeDistributed, Dense, Dropout, MaxPooling2D
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.applications import InceptionResNetV2
# from keras.layers.wrappers import TimeDistributed
from tensorflow.keras.layers import TimeDistributed
# from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D, MaxPooling2D)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers

from collections import deque
import sys
import os
# from lstm_extract_features import extract_features
import time
import pandas as pd
import numpy as np
from pytvid import get_all_sequences_in_memory, frame_generator
from sklearn.model_selection import train_test_split
from lstm_extract_features import load_npy_features

# length of train and test sets
df_train = pd.read_csv('train_df.csv')
df_test = pd.read_csv('test_df.csv')
len_train = len(df_train)
len_test = len(df_test)
# print (len_train,'\n',len_test)
# sys.exit()

class LSTM_Architecture:
    def __init__(self, model, nb_classes=2, seq_length=10, saved_model=None, features_length=2048):
        self.nb_classes = nb_classes
        self.saved_model = saved_model
        self.seq_length = seq_length
        self.model = model
        self.feature_length = features_length

        metrics = ['accuracy']
        if self.saved_model is not None:
            print ('Loading model {}'.format(self.saved_model))
            self.model =load_model(self.saved_model)
        elif model == 'lstm':
            print ('Loading LSTM model...')
            self.input_shape = (seq_length, features_length)
            self.model = self.timeModel()
        else:
            print ('Unknown network')
            sys.exit()

        # now the model has to be ccompiled
        optimizer = Adam(lr=1e-5, decay=1e-6)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
        print (self.model.summary())

    def lstm(self):
        model = Sequential()
        model.add(
            LSTM(2048, return_sequences=False,
                    input_shape=self.input_shape,
                 dropout=.5)
        )
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))
        return model

    def my_base_model(self, input_shape=(299, 299, 3)):
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
        return model

    def timeModel(self, input_shape=(10,299,299,3)):
        convnet = self.my_base_model(input_shape[1:])

        model = Sequential()
        model.add(TimeDistributed(convnet, input_shape=input_shape))
        model.add(LSTM(64))

        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        return model

    def myLSTM(self, input_shape=(224, 224, 3)):
        googleNet_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        googleNet_model.trainable = True

        model = Sequential()
        model.add(TimeDistributed(googleNet_model))
        model.add(Flatten())
        model.add(LSTM(units=10))
        return model

    def train_model(self):
        from keras_video import VideoFrameGenerator
        from keras.preprocessing.image import ImageDataGenerator
        from keras_video.utils import show_sample
        classes = ['Real', 'Deepfake']
        classes.sort()

        size = (224, 224)
        channels = 3
        nbframe = 10
        bs = 2

        glob_pattern = 'Dataset_Face_Extracted_redo/{classname}/*.mp4'

        data_aug = ImageDataGenerator(
            zoom_range=.1,
            horizontal_flip=True,
            rotation_range=8,
            width_shift_range=.2,
            height_shift_range=.2)

        train_generator = VideoFrameGenerator(
            classes=classes,
            glob_pattern=glob_pattern,
            nb_frames=nbframe,
            split=.25,
            shuffle=True,
            batch_size=bs,
            target_shape=size,
            nb_channel=channels,
            transformation=data_aug,
            use_frame_cache=True
        )
        valid = train_generator.get_validation_generator()
        show_sample(train_generator)

        input_shape = (nbframe,) + size + (channels,)
        model = self.timeModel(input_shape)
        model.compile(Adam(0.001), 'categorical_crossentropy', metrics=['acc'])

        epochs = 50
        callbacks = [
            ReduceLROnPlateau(verbose=1),
            ModelCheckpoint(
                filepath=os.path.join('dataLSTM', 'checkpoints', str(model) + \
                                      '.{epoch:03d}-{val_loss:.3f}.hdf5'),
                verbose=1, save_best_only=True
            )
        ]
        model.fit_generator(
          train_generator, validation_data=valid, verbose=1, epochs=epochs, callbacks=callbacks
        )

mm = LSTM_Architecture('lstm').myLSTM()


sys.exit()

def train(data_type, model, seq_length=10, saved_model=None, class_limit=2, image_shape=None,
          load_to_memory=False, nb_epoch=10, batch_size=32):
    # df_train = pd.read_csv('train_df.csv', converters={'video_file_name': lambda x: str(x)})
    # df_test = pd.read_csv('test_df.csv', converters={'video_file_name': lambda x: str(x)})
    df = pd.read_csv('dataLSTM/df_dataLSTM.csv', converters={'video_file_name': lambda x: str(x)})

    checkpointer = ModelCheckpoint(
        filepath=os.path.join('dataLSTM', 'checkpoints', str(model) + '-' + str(data_type) + \
                              '.{epoch:03d}-{val_loss:.3f}.hdf5'),
        verbose=1, save_best_only=True
    )
    # print(model)
    tb = TensorBoard(log_dir=os.path.join('dataLSTM', 'logs', str(model)))
    early_stopper = EarlyStopping(patience=2)

    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('dataLSTM', 'logs', model + '-' + 'training-' + \
                                        str(timestamp) + '.log'))

    # if image_shape is None: data = Dataset(seq_length=seq_length, class_limit=class_limit)
    # else: data = Dataset(seq_length=seq_length, class_limit=class_limit, image_shape=image_shape)

    steps_per_epoch = (len_train*.7)//batch_size

    if load_to_memory:
        x, y = load_npy_features(df) # get_all_sequences_in_memory(df,'file_pth')
        y = to_categorical(y,2)

        trainX, valX, trainY, valY = train_test_split(x, y, train_size=0.9, test_size=0.1, random_state=42)
        x_test, y_test = load_npy_features(df, train_test='test') # get_all_sequences_in_memory(df_test)
    else:
        print('Loading generators...')
        train_generator = frame_generator(df,'train',data_type)
        test_generator = frame_generator(df, 'test', data_type)

    rm = LSTM_Architecture('lstm',features_length=64)

    if load_to_memory:
        rm.model.fit(trainX, trainY, batch_size=10,
                     validation_data=(valX,valY),
                     callbacks=[tb,early_stopper,csv_logger,checkpointer], epochs=nb_epoch, verbose=1)
    else:
        rm.model.fit_generator(generator=train_generator,
                               steps_per_epoch=steps_per_epoch,
                               callbacks=[tb,early_stopper,csv_logger,checkpointer], epochs=nb_epoch,
                               verbose=1, workers=4)

def main():
    sequences_dir = os.path.join('dataLSTM', 'data')
    if not os.path.exists(sequences_dir):
        os.mkdir(sequences_dir)

    checkpoints_dir = os.path.join('dataLSTM', 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    image_height, image_width = 299, 299

    # model can be only 'lstm'
    model = 'lstm'
    saved_model = None  # None or weights file
    load_to_memory = False  # pre-load the data into memory
    batch_size = 1
    nb_epoch = 100
    data_type = 'data'
    image_shape = (image_height, image_width, 3)
    seq_length = 1
    class_limit = 2

    # extract_features('dataLSTM/df_dataLSTM.csv', seq_length=seq_length, class_limit=class_limit, image_shape=image_shape)
    # train(data_type=data_type, seq_length=seq_length, model=model, saved_model=saved_model,
    #       class_limit=class_limit, image_shape=image_shape,
    #       load_to_memory=load_to_memory, batch_size=batch_size, nb_epoch=nb_epoch)

    lstm = LSTM_Architecture('lstm',features_length=64)
    lstm.train_model()

if __name__ == '__main__':
    main()
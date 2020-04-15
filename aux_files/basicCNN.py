from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, BatchNormalization, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Conv2D, MaxPooling3D, Conv3D,MaxPooling2D
from collections import deque
import sys
import warnings
import os
warnings.filterwarnings("ignore")

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.xception import Xception, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from keras.applications.vgg16 import VGG16
from keras.layers import Input, GlobalAveragePooling2D
from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical
import tensorflow as tf

# tf.logging.set_verbosity(tf.logging.ERROR)

# other imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import glob
# from PIL import load_img
import cv2
import h5py
import os
import json
import datetime
import time
import matplotlib.pyplot as plt


# img = load_img('data/validation/Deepfake/0.jpg')
# img = img_to_array(img)
# print(img.shape)
# sys.exit()
def load_data_dir(batch_size=32):
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        'data' + os.path.sep + 'train',
        target_size=(100,100),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        'data' + os.path.sep + 'validation',
        target_size=(100,100),
        batch_size=batch_size,
        class_mode='binary')

    return train_generator, validation_generator

def load_data_flow(batch_size=32, test_df=False):
    df = pd.read_csv('my_df_org.csv', header=0)
    df = df[df.columns[1:]]
    msk = np.random.rand(len(df)) < 0.75
    train = df[msk]
    test = df[~msk]
    if test_df: return test
    # print(test.columns)
    # sys.exit()

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.25)
    train_generator = datagen.flow_from_dataframe(
        dataframe = train,
        directory = None, #'data' + os.path.sep + 'train',
        target_size=(100,100),
        batch_size=batch_size,
        class_mode='binary',
        x_col = 'file',
        y_col = 'label',
        seed=42,
        shuffle=True,
        subset = 'training')

    validation_generator = datagen.flow_from_dataframe(
        dataframe=train,
        directory = None, #'data' + os.path.sep + 'validation',
        target_size=(100,100),
        batch_size=batch_size,
        class_mode='binary',
        x_col = 'file',
        y_col = 'label',
        seed=42,
        shuffle=True,
        subset = 'validation')

    test_datagen = ImageDataGenerator(rescale=1./255.)
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test,
        directory = None, #'data' + os.path.sep + 'validation',
        target_size=(100,100),
        batch_size=batch_size,
        class_mode=None,
        x_col = 'file',
        y_col = 'label',
        seed=42,
        shuffle=False)

    return train_generator, validation_generator, test_generator
# load_data_flow()
# sys.exit()

def train_architecture(model, batch_size=32, epochs=100):
    train_generator, validation_generator = load_data_dir()
    early_stopping = EarlyStopping(patience=4)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=800 // batch_size,
        callbacks = [early_stopping])
    model.save_weights('my_weightsvgg.h5')
    model_json = model.to_json()
    with open('model_vgg.json', 'w') as json_file:
        json_file.write(model_json)

def load_dataset(df_name, batch_size=32, mode='train'):
    df = pd.read_csv(df_name)
    df = pd.DataFrame(df.head(round(len(df)*.75)))
    labels_classes = list(set(df.label))
    data = []
    labels = []
    # print (labels_classes)
    # sys.exit()
    for i in range(len(df['file'])):
        file_dir = df['file'].iloc[i]
        img = cv2.imread(file_dir)
        if img is not None:
            img = cv2.resize(img, (128,128))
            img = img_to_array(img)
            data.append(img)

            # print(df['label'].iloc[i])
            label = 0 if df['label'].iloc[i] is labels_classes[0] else 1
            # print (label)
            labels.append(label)
    data = np.array(data, dtype=np.float32)/255.0
    labels = np.array(labels)

    if mode is 'train':
        trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.2, random_state=42)
        # trainY = to_categorical(trainY)
        # testY  = to_categorical(testY)
        return trainX, testX, trainY, testY
    else:
        return data, labels


def train_architecture_df(model, batch_size=32, epochs=100):
    trainX, testX, trainY, testY = load_dataset(df_name='train_df.csv')
    # print(trainX.shape)
    # for i in range(12):
    #     print(trainY[i])
    # print(type(trainX))
    # print(type(trainX[0]))
    # print (set(trainY))
    # sys.exit()
    early_stopping = EarlyStopping(patience=2)

    STEP_SIZE_TRAIN = len(trainX) // batch_size
    # STEP_SIZE_VALID = validati // validation_generator.batch_size
    STEP_SIZE_TEST = len(testX) // batch_size

    img_generator = ImageDataGenerator()

    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(),
                  metrics=['accuracy'])

    # H = model.fit_generator(img_generator.flow(trainX, trainY, batch_size=batch_size),
    #           validation_data=(testX, testY), steps_per_epoch=STEP_SIZE_TRAIN,
    #           epochs=epochs, verbose=1, callbacks = [early_stopping],
    #               validation_steps=STEP_SIZE_TEST)
    H = model.fit(trainX, trainY, batch_size=batch_size,
                  validation_data=(testX,testY),
              epochs=epochs, verbose=1, callbacks = [early_stopping])
    model.save_weights('my_weightsvgg-DF.h5')
    model_json = model.to_json()
    with open('model_vgg-DF.json', 'w') as json_file:
        json_file.write(model_json)

    # plt.style.use("ggplot")
    # plt.figure()
    # N = epochs
    # plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    # plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    # plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    # plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    # plt.title("Training Loss and Accuracy")
    # plt.xlabel("Epoch #")
    # plt.ylabel("Loss/Accuracy")
    # plt.legend(loc="lower left")
    # plt.savefig('acc.png')


def load_architecture(json_file, h5_file):
    json_file = open(json_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(h5_file)
    print('success')
    return loaded_model

def evaluate_model(model):
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    train_generator, validation_generator = load_data_dir()
    step_size_valid = validation_generator.n//validation_generator.batch_size
    score = model.evaluate_generator(generator = validation_generator, steps = step_size_valid)
    return score

def evaluate_model_df(model, batch_size=32):
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    # test_datagen = ImageDataGenerator()
    # test = pd.read_csv('test_df.csv')
    # test_generator = test_datagen.flow_from_dataframe(
    #     dataframe=test,
    #     directory = None, #'data' + os.path.sep + 'validation',
    #     target_size=(100,100),
    #     batch_size=batch_size,
    #     class_mode=None,
    #     x_col = 'file',
    #     y_col = 'label',
    #     seed=42,
    #     shuffle=False)

    data, labels = load_dataset(df_name='test_df.csv', mode='test')
    # for l in labels: print(l, '\t', end='')
    labels = to_categorical(labels)
    # for l in labels: print(l, '\t', end='')
    # print (labels[0].shape)
    # sys.exit()

    # step_size_test = test_generator.n//test_generator.batch_size
    score = model.evaluate(data, labels, batch_size=batch_size)
    # score = model.evaluate_generator(test_generator, steps = step_size_test)
    return score


# model = load_architecture('model_vgg-DF.json', 'my_weightsvgg-DF.h5')
import mytrain
model = mytrain.ggnModel()
model.load_weights('my_weights.h5')
score = evaluate_model_df(model)
print(model.summary())
print(score,'\ntype: ',type(score))
# print(len(score))
# for o in np.rint(score): print(o)
# y_classes = np.argmax(score, axis=-1)
# print(y_cla/sses)
# print(model.metrics_names)
sys.exit/()

# df=load_data_flow(test_df=True)
# # print(df.columns)
# n=18999
# print(df['label'].iloc[n])
# tst_img = load_img(df['file'].iloc[n])
# tst_img = img_to_array(tst_img)
# tst_img = np.expand_dims(tst_img, axis=0)
# result = model.predict_proba(tst_img)
# print(result)
#
# print(df[df['label'] == 'Real'])
# sys.exit()

def predict(model):
    train_generator, validation_generator, test_generator = load_data_flow()
    step_size_test = test_generator.n // test_generator.batch_size
    test_generator.reset()
    pred = model.predict_generator(test_generator, steps=step_size_test, verbose=1)
    predicted_class_indices = np.argmax(pred, axis=1)

    labels = (train_generator.class_indices)
    labels = dict((v, k) for k, v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]

    filenames = test_generator.filenames
    results = pd.DataFrame({"Filename": filenames,
                            "Predictions": predictions})
    results.to_csv("results.csv", index=False)


# plot_model(model, to_file='model_cnn2d.jpg', show_shapes=True, show_layer_names=True)

def cnn2d(input_shape = (100, 100, 3)):
    classifier = Sequential()
    classifier.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), input_shape=input_shape, activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    classifier.add(BatchNormalization())
    classifier.add(Dropout(0.6))

    classifier.add(Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), input_shape=input_shape, activation='relu',
                          padding='same'))
    classifier.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    classifier.add(BatchNormalization())
    classifier.add(Dropout(0.6))

    classifier.add(Flatten())
    classifier.add(Dense(units=32, activation='relu'))
    classifier.add(Dense(units=1, activation='sigmoid'))
    return classifier
# train_architecture(cnn2d())
# sys.exit()

def vgg_architecture(input_shape = (100, 100, 3)):
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # for layer in vgg.layers[:10]:layer.trainable = False
    # for layer in vgg.layers:
    #     sp = '         '[len(layer.name)-9:]
    #     print(layer.name, sp, layer.trainable)
    x = vgg.output
    # x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    # x = Dense(2, activation='sigmoid')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=vgg.input, outputs=x)
    return model
# train_architecture_df(vgg_architecture(), epochs=10)

# loaded_model = load_architecture('model_vgg.json','my_weightsvgg.h5')
# print(loaded_model.summary())
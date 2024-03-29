"""
A collection of models we'll use to attempt to classify videos.
"""
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, MaxPooling2D, Bidirectional, GRU
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras_vggface.vggface import VGGFace
# from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D, MaxPooling2D)
from collections import deque
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
import sys

# modell = VGGFace(model='resnet50')
# print(modell.summary())
# sys.exit)

class ResearchModels():
    def __init__(self, nb_classes, model, seq_length,
                 saved_model=None, features_length=2622):
        """
        `model` = lstm (only one for this case)
        `nb_classes` = the number of classes to predict
        `seq_length` = the length of our video sequences
        `saved_model` = the path to a saved Keras model to load
        """

        # Set defaults.
        self.seq_length = seq_length
        self.load_model = load_model
        self.saved_model = saved_model
        self.nb_classes = nb_classes
        self.feature_queue = deque()

        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy']
        if self.nb_classes >= 10:
            metrics.append('top_k_categorical_accuracy')

        # Get the appropriate model.
        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)
            # self.model = load_model(self.saved_model)
            self.input_shape = (seq_length, features_length)
            self.model = self.lstm()
            # self.model.load_weights(saved_model)
        elif model == 'lstm':
            print("Loading LSTM model.")
            self.input_shape = (seq_length, features_length)
            print('input shape: {}'.format(self.input_shape))
            self.model = self.lstm()
        elif model == 'bilstm':
            print("Loading Bi-LSTM model.")
            self.input_shape = (seq_length, features_length)
            print('input shape: {}'.format(self.input_shape))
            self.model = self.bilstm()
        elif model == 'gru':
            print("Loading gru model.")
            self.input_shape = (seq_length, features_length)
            print('input shape: {}'.format(self.input_shape))
            self.model = self.gru()
        else:
            print("Unknown network.")
            sys.exit()

        # Now compile the network.
        optimizer = Adam(lr=1e-5, decay=1e-6)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                           metrics=metrics)
        # print(self.model.summary())

    def lstm(self):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predomenently."""
        # Model.
        model = Sequential()
        model.add(LSTM(2048, return_sequences=False,
                       input_shape=self.input_shape,
                       dropout=0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))
        # print(model.summary())
        # print(self.input_shape)
        # sys.exit()

        return model

    def bilstm(self):
        """Build a simple BiLSTM network. We pass the extracted features from
        our CNN to this model predomenently."""
        # Model.
        model = Sequential()
        model.add(Bidirectional(LSTM(2048, return_sequences=False,
                       input_shape=self.input_shape,
                       dropout=0.5)))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))
        # print(model.summary())
        # print(self.input_shape)
        # sys.exit()

        return model
    
    def gru(self):
        """Build a simple GRU network. We pass the extracted features from
        our CNN to this model predomenently."""
        # Model.
        model = Sequential()
        model.add(GRU(2048, return_sequences=False,
                       input_shape=self.input_shape,
                       dropout=0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))
        return model


# lsym = ResearchModels(model='bilstm',seq_length=10, nb_classes=2).model
# lsym.build()
# print(lsym.summary())
# from keras.utils.vis_utils import plot_model
# plot_model(lsym,'lstm.jpg', expand_nested=True, show_shapes=True)

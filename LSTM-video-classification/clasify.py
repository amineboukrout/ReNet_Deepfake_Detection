import os
import sys
import cv2
import numpy as np
from data import DataSet
from extractor import Extractor
from keras.models import load_model
from models import ResearchModels
import sys
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

# if (len(sys.argv) == 5):
#     seq_length = int(sys.argv[1])
#     class_limit = int(sys.argv[2])
#     saved_model = sys.argv[3]
#     video_file = sys.argv[4]
# else:
#     print ("Usage: python clasify.py sequence_length class_limit saved_model_name video_file_name")
#     print ("Example: python clasify.py 75 2 lstm-features.095-0.090.hdf5 some_video.mp4")
#     exit (1)


seq_length = 20
saved_model = 'cnn_lstm_VGGFace10_SPLIT0.h5'
video_file = 'data/test/Deepfake/46.mp4'

def classify(video_file, seq_length=20, saved_model = './cnn_lstm_VGGFace10.h5'):
    capture = cv2.VideoCapture(os.path.join(video_file))
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    print('#########################################################',video_file,'#########################################################')

    # Get the dataset.
    data = DataSet(seq_length=seq_length, class_limit=2, image_shape=(224, 224, 3))

    # get the model.
    extract_model = Extractor(image_shape=(height, width, 3))
    rm = ResearchModels(len(data.classes), 'lstm', seq_length, saved_model, features_length=2622)
    saved_LSTM_model = rm.lstm()
    saved_LSTM_model.load_weights(saved_model)

    frames = []
    frame_count = 0
    while True:
        ret, frame = capture.read()
        print(ret)
        # Bail out when the video file ends
        if not ret:
            break

        # Save each frame of the video to a list
        frame_count += 1
        frames.append(frame)

        if frame_count < seq_length:
            continue # capture frames untill you get the required number for sequence
        else:
            frame_count = 0

        # For each frame extract feature and prepare it for classification
        sequence = []
        for image in frames:
            image = cv2.resize(image, (224,224), 3)
            features = extract_model.extract_image(image)
            sequence.append(features)

        # Clasify sequence
        prediction = saved_LSTM_model.predict(np.expand_dims(sequence, axis=0))
        print('classofyyyyyyyyyyy')
        print(prediction)
        values = data.print_class_from_prediction(np.squeeze(prediction, axis=0))
        # print(np.argmax(prediction))

        frames = []
    print(np.argmax(prediction))
    return np.argmax(prediction)

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
    plt.savefig('cm_split4')
    return cm

# df = pd.read_csv('data/data_file.csv', converters={'file': lambda x: str(x)})
# df.columns = ['split','class','file','framesNo']
# df = pd.DataFrame(df[df.split == 'test'])
# df = df[df['class']=='Deepfake']

# print(classify(video_file, saved_model=saved_model))

'''one_HOT = []
predictions = []

for _,row in df.iterrows():
    if row['class'] == 'Deepfake':
        one_HOT.append(0)
        print('deeeeeeeeeeeeepfake')
    else: one_HOT.append(1)

    path = os.path.join('data',row['split'],row['class'],str(row['file'])+'.mp4')
    predictions.append(classify(path))

for i in range(len(one_HOT)):
    print(one_HOT[i],'\t',predictions[i])

print_confusion_matrix(one_HOT,predictions)'''


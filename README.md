# ReNet: Deepfake Detection

## Description
This project is for my Masters' thesis at the University of Warwick. It is based on the field of Digital Image Forensics using a combination of Computer Vision and Machine Learning techniques.

## Process summary
I am in the process of developing various Python models that determines whether a given video is a Deepfake (i.e. altered). The current process is:

 - Data Preparation - face extraction, splitting videos into fixed length (i.e. 1 second), split data into a 80-20 train/test split
 - Feature extraction - feed a video, frame-by-frame, into a pre-trained Constitutional Neural Network (CNN). Features are extracted from a specified layer from the CNN.
 - Feed extracted features into a recurrent model - the recurrent model is trained as a binary classification model on the extracted features from the CNN.
 - Using the full model, CNN-recurrent, to make a prediction on a video.
 - Result Analysis using Metrics, such as: accuracy, confusion matrix, F1 Score, True Positive Rate, etc.
## Preliminary accuracy
Average accuracy is 94%.

Note, the code is currently not structured, this will be rectified in due course along with a more detailed documentation.
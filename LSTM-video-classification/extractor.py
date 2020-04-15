from keras.preprocessing import image
# from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Input
import numpy as np
# from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
import sys
from keras_vggface.vggface import VGGFace
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, MaxPooling2D


class Extractor():
    def __init__(self, image_shape=(299, 299, 3), weights=None):
        """Either load pretrained from imagenet, or load our saved
        weights from our own training."""

        self.weights = weights  # so we can check elsewhere which model

        input_tensor = Input(image_shape)
        # Get model with pretrained weights.
        # base_model = InceptionV3(
        #     input_tensor=input_tensor,
        #     weights='imagenet',
        #     include_top=True
        # )

        # base_model = InceptionResNetV2(
        #     input_tensor=input_tensor,
        #     weights='imagenet',
        #     include_top=True
        # ) # 2622

        base_model = VGGFace()
        # print(base_model.summary())
        # sys.exit()

        # We'll extract features at the final pool layer.
        self.model = Model(
            inputs=base_model.input,
            outputs=base_model.get_layer('fc8').output # vggface use fc8, rest avg_pool
        )
        # print(self.model.summary())
        # print(base_model.get_layer('fc8').output.shape)
        # sys.exit()

    def extract(self, image_path):
        img = image.load_img(image_path, target_size=(224,224,3)) #<-- for vggface
        # print(img.shape)
        # sys.exit()
        return self.extract_image(img)

    def extract_image(self, img):
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        # x = preprocess_input(x)

        # Get the prediction.
        # print(x.shape)
        features = self.model.predict(x)

        if self.weights is None:
            # For imagenet/default network:
            features = features[0]
        else:
            # For loaded network:
            features = features[0]

        return features

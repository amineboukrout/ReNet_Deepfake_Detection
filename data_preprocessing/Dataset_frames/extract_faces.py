# source: https://github.com/kb22/Create-Face-Data-from-Images

import sys
import cv2
import os
import glob
import numpy as np

def data_generator(new_dataset_name='updated_images'):
    base_dir = os.path.dirname(__file__)
    prototxt_path = os.path.join(base_dir, 'model_data', 'deploy.prototxt')
    caffemodel_path = os.path.join(base_dir, 'model_data', 'weights.caffemodel')

    model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

    if not os.path.exists(new_dataset_name):
        os.mkdir(new_dataset_name)
    if not os.path.exists(new_dataset_name + os.path.sep + 'Deepfake'):
        os.mkdir(new_dataset_name + os.path.sep + 'Deepfake')
    if not os.path.exists(new_dataset_name + os.path.sep + 'Real'):
        os.mkdir(new_dataset_name + os.path.sep + 'Real')

    cur_folders = ['Deepfake', 'Real']
    print('here')

    for clas in cur_folders:
        image_folders = os.listdir(clas)
        print(image_folders)
        for image_folder in image_folders:
            images = glob.glob(os.path.join(clas, image_folder, '*.jpg')) # os.listdir(os.path.join(clas, image_folder))
            os.mkdir(os.path.join(new_dataset_name,clas,image_folder))
            save_image = False
            for image in images:
                # print(os.path.join(image))
                # os.mkdir(image)
                # sys.exit()
                image_file = cv2.imread(os.path.join(image))
                (h, w) = image_file.shape[:-1]
                blob = cv2.dnn.blobFromImage(cv2.resize(image_file, (300,300)), 1.0, (300,300), (104, 177, 123))

                model.setInput(blob)
                detections = model.forward()

                # create frame around face
                for i in range(0, detections.shape[2]):
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    confidence = detections[0, 0, i, 2]
                    # print('8hhhhhhhhhhhhhhh')

                    # if confidence > 0.5, show box around face
                    if confidence > 0.5:
                        save_image = True
                        cv2.rectangle(image_file, (startX, startY), (endX, endY), (255, 255, 255), 2)
                if save_image:
                    cv2.imwrite(os.path.join(base_dir,new_dataset_name, image), image_file)
                    print('Image '+ image +' converted successfully')
                else: print('no face')
            print('Done with image!!!!!!!!!!!')

def extract_faces(new_dataset_name='face_images'):
    base_dir = os.path.dirname(__file__)
    prototxt_path = os.path.join(base_dir, 'model_data', 'deploy.prototxt')
    caffemodel_path = os.path.join(base_dir, 'model_data', 'weights.caffemodel')

    model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

    if not os.path.exists(new_dataset_name):
        os.mkdir(new_dataset_name)
    if not os.path.exists(new_dataset_name + os.path.sep + 'Deepfake'):
        os.mkdir(new_dataset_name + os.path.sep + 'Deepfake')
    if not os.path.exists(new_dataset_name + os.path.sep + 'Real'):
        os.mkdir(new_dataset_name + os.path.sep + 'Real')

    cur_folders = ['Deepfake', 'Real']
    count = 0

    for clas in cur_folders:
        image_folders = os.listdir(clas)
        print(image_folders)
        for image_folder in image_folders:
            images = glob.glob(os.path.join(clas, image_folder, '*.jpg')) # os.listdir(os.path.join(clas, image_folder))
            os.mkdir(os.path.join(new_dataset_name,clas,image_folder))
            save_image = False
            for image in images:
                # print(os.path.join(image))
                # os.mkdir(image)
                # sys.exit()
                image_file = cv2.imread(os.path.join(image))
                (h, w) = image_file.shape[:-1]
                blob = cv2.dnn.blobFromImage(cv2.resize(image_file, (300,300)), 1.0, (300,300), (104, 177, 123))

                model.setInput(blob)
                detections = model.forward()

                # create frame around face
                for i in range(0, detections.shape[2]):
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    confidence = detections[0, 0, i, 2]
                    # print('8hhhhhhhhhhhhhhh')

                    # if confidence > 0.5, show box around face
                    if confidence > 0.5:
                        if i == 0:
                            count += 1
                            frame = image_file[startY:endY, startX:endX]
                            try:
                                cv2.imwrite(os.path.join(base_dir, new_dataset_name, image), frame)
                                print  ('Wrote: {}'.format(os.path.join(base_dir, new_dataset_name, image)))
                            except:
                                print('Couldn\'t write: {}'.format(os.path.join(base_dir, new_dataset_name, image)))
                            # print(os.path.join(base_dir, new_dataset_name, image.split('.')[0] + '_frame'+str(i)) + '.' + image.split('.')[1])
                            # print('heeeeeeeeeeeeeeere')
                            # sys.exit()
                            # cv2.imwrite(os.path.join(base_dir, new_dataset_name, image.split('.')[0] + '_frame'+str(i) + '.' + image.split('.')[1]), frame)
            print('Done with a video!')

# extract_faces()
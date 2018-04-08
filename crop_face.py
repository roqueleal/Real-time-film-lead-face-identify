# -*- coding:utf-8 -*-
# ! ./usr/bin/env python
# __author__ = 'zzp'

# Use MTCNN to detect face

import tensorflow as tf
import numpy as np
import cv2
import os
import detect_face

# hyper parameters
minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def load_model():
    print('Creating networks and loading parameters')
    gpu_memory_fraction = 1.0
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, './model_check_point/')

    return pnet, rnet, onet


def corp_faces(images, image_path, save_path, pnet, rnet, onet):
    """
    Use MTCNN crop faces
    :param images: a list, all images for this 'people'
    :param save_path: where to save crop face
    :return:
    """
    count = 0
    for image_name in images:
        print('processing ' + image_path + '/' + image_name)
        frame = cv2.imread(image_path + '/' + image_name)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if gray.ndim == 2:
            img = to_rgb(gray)

        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

        nrof_faces = bounding_boxes.shape[0]  # number of faces
        print('Total faces：{}'.format(nrof_faces))

        # crop and save
        for face_position in bounding_boxes:
            face_position = face_position.astype(int)
            crop = frame[face_position[1]:face_position[3], face_position[0]:face_position[2], :]

            cv2.imwrite(save_path + '/' + (str(count) + '.jpg'), crop)
            count += 1


if __name__ == '__main__':

    image_path = './images'
    names = os.listdir(image_path)
    print('Total ' + str(len(names)) + 'people')

    # load model
    pnet, rnet, onet = load_model()

    for index, name in enumerate(names):
        cur_people = name

        # create dir for save people face
        save_path = './faces/' + cur_people
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # image names for this people
        imgs = os.listdir(image_path + '/' + cur_people)

        # crop image and save
        corp_faces(imgs, './images/'+cur_people, save_path, pnet, rnet, onet)


# -*- coding:utf-8 -*-
# ! ./usr/bin/env python
# __author__ = 'zzp'

# Use vgg_face2 extract face feature

import os
import pickle
import numpy as np
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils


def pickle_stuff(filename, stuff):
    save_stuff = open(filename, "wb")
    pickle.dump(stuff, save_stuff)
    save_stuff.close()


def image2x(image_path):
    # convert image format for keras
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_input(x, version=1)  # or version=2
    return x


def to_batch(lead_dir, lead_img):
    """
    make lead faces to batch for vgg_face
    :param lead_dir: which people
    :param lead_img: a list, lead images
    :return:
    """
    for index, img_name in enumerate(lead_img):
        x = image2x(lead_dir + '/' + img_name)
        if index == 0:
            batch = x
        else:
            batch = np.concatenate((batch, x), axis=0)

    return batch


if __name__ == '__main__':
    face_dir = './faces/'
    names = os.listdir(face_dir)
    print('Total ' + str(len(names)) + 'people')

    # build model
    resnet50_features = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

    lead_feats = {}
    for name in names:
        print('processing ' + name)
        lead_dir = face_dir + name

        lead_img = os.listdir(lead_dir)

        x = to_batch(lead_dir, lead_img)

        # reference
        batch_feats = resnet50_features.predict(x)
        # mean
        lead_feats[name] = np.mean(batch_feats, axis=0)

    # save lead feature tp .pkl
    pkl_name = 'lead_feats.pkl'
    pickle_stuff(pkl_name, lead_feats)








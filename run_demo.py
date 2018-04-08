# -*- coding:utf-8 -*-
# ! ./usr/bin/env python
# __author__ = 'zzp'

# This is a demo for find lead in 'Friends', I chose session3 1st, you can change to others

import cv2
import pickle
import detect_face
import numpy as np
from scipy import spatial
from keras_vggface.vggface import VGGFace
from crop_face import load_model, to_rgb

# hyper parameters for mtcnn
minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor


def load_stuff(pkl_name):
    saved_stuff = open(pkl_name, "rb")
    stuff = pickle.load(saved_stuff)
    saved_stuff.close()
    return stuff


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point  # left-up corner
    new_img = cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    new_img = cv2.putText(new_img, label, point, font, font_scale, (255, 255, 255), thickness)
    
    return new_img


class FaceIdentify(object):
    # class for FaceIdentify
    def __init__(self, precompute_lead_face=None):
        self.face_size = 224
        self.precompute = load_stuff(precompute_lead_face)
        print('Loading MTCNN model...')
        self.pnet, self.rnet, self.onet = load_model()
        print('MTCNN have been loaded!...')
        print('Loading VGG Face model...')
        self.vgg_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        print('VGG_Face have been loaded!...')

    def identify_face(self, features, threshold=120):
        min_distance = float('inf')
        for lead_name in self.precompute.keys():
            lead_features = self.precompute[lead_name]
            distance = spatial.distance.euclidean(lead_features, features)
            if distance < min_distance:
                min_distance = distance
                lead = lead_name

        if min_distance < threshold:
            return lead
        else:
            return 'Others'

    def detect_face(self, video_path):
        cap = cv2.VideoCapture(video_path)

        # save video
        # Define the codec and create VideoWriter object
        out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*"XVID"), 20.0, (720, 400))

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if gray.ndim == 2:
                  img = to_rgb(gray)

                # crop face
                bounding_boxes, _ = detect_face.detect_face(img, minsize, self.pnet, self.rnet, self.onet, threshold, factor)
                face_imgs = np.empty((len(bounding_boxes), self.face_size, self.face_size, 3))

                if bounding_boxes.shape[0]:
                    for i, face_position in enumerate(bounding_boxes):
                        face_position = face_position.astype(int)
                        crop = frame[face_position[1]:face_position[3], face_position[0]:face_position[2], :]
                        if crop.shape[0] and crop.shape[1]:
                            crop_resize = cv2.resize(crop, (self.face_size, self.face_size), interpolation=cv2.INTER_AREA)
                            face_imgs[i, :, :, :] = crop_resize
                            cv2.rectangle(frame, (face_position[0], face_position[1]),
                                          (face_position[2], face_position[3]), (0, 255, 0), 2)


                # vgg_face2 features
                if len(face_imgs) > 0:
                    features_faces = self.vgg_model.predict(face_imgs)
                    predicted_names = [self.identify_face(features_face) for features_face in features_faces]

                # draw results
                if bounding_boxes.shape[0]:
                    for index, face in enumerate(bounding_boxes):
                      label = "{}".format(predicted_names[index])
                      save_frame = draw_label(frame, (int(face[0]), int(face[1])), label)
                      out.write(save_frame)  # save
                else:
                    out.write(frame)  # save

                #cv2.imshow('Friends', frame)
                
                #if cv2.waitKey(5) == 27:  # ESC key press
                  #break
            else:
                break

        # When everything is done, release the capture
        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':

    # read video
    video_path = './friends0301_cut.mp4'

    face = FaceIdentify(precompute_lead_face="./lead_feats.pkl")
    face.detect_face(video_path=video_path)





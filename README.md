---
layout: post
title: Real Time Film-Lead Face Identify
date: 2018-04-08 10:50:24.000000000 +09:00
tags: Face Detection
comments: true
---

### **About**
Since I love [Friends of six](https://zh.wikipedia.org/wiki/%E8%80%81%E5%8F%8B%E8%AE%B0) so much, I decide to make a demo for identifying their faces in the video. BTW, the demo is naive, you can make more effort on this for a better result. And `real time` means on a good GPU rather than a bad PC, since two CNN take a while. 

[![test_save.jpg](https://s20.postimg.org/i5avy6anx/test_save.jpg)](https://postimg.org/image/xe0tby4c9/)

### **Pipeline**
The pipeline of my work is easy understood.<br/>
 First, I collect 10 images for the six lead. `Rachel` `Monica` `Phoebe` `Ross` `Joey` `Chandler`. <br/>
 Then, I use [MTCNN](https://arxiv.org/abs/1604.02878) to detect face, and crop these faces to save them. One of my [reference](https://www.dlology.com/blog/live-face-identification-with-pre-trained-vggface2-model/), it uses `opencv` built-in function to finish this, it's fast but less effective in tough condition.<br/>
 After that, [VGG_face2](https://arxiv.org/abs/1710.08092) is used to get features for each people, and save it to `.pkl` format.<br/>
 Finally, a `run demo` is made to combine all above. For simplicity, I use `Euclidean Distance` here. You may change it to other metric way like `cosine distance`.

[![image](https://github.com/JudasDie/Real-time-film-lead-face-identify/output.png)](https://www.youtube.com/watch?v=8yf12Pq379c)

 I will show you some details.<br/>

### **MTCNN for face detection**

 I have shown three ways for face detection in my previous [article](https://judasdie.github.io/2018/04/three-ways-for-face-detection/). 
 ``` python
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
 ```
 Before using it, please `clone` my [github](https://github.com/JudasDie/Real-time-film-lead-face-identify). Or you can download the file in directory `model_checkpoint` only, and modify my code.

### **Use vgg_face2 get face features**
 [VGG_face2](https://arxiv.org/abs/1710.08092) is a higher version of `vgg face`. You can choose base net of `vgg` `resnet`. Gratefully, `rcmalli` have made a `keras` version for vgg face. Vgg face will generate a 2048 dimentional vector.
 ```
pip install keras_vggface
 ```
 Here is an official demo.
 ```python
import numpy as np
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils

model = VGGFace() # default : VGG16 , you can use model='resnet50' or 

img = image.load_img('../image/ajb.jpg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = utils.preprocess_input(x, version=1) # or version=2
preds = model.predict(x)
print('Predicted:', utils.decode_predictions(preds))
 ```

### **Combine**
After done tow steps above, you almost get there. The thing lefted is reading video and doing the `pipeline` above. I really suggest you try other metric ways.
```python
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
```

### **How to use my repo**
```
>>>git clone https://github.com/JudasDie/Real-time-film-lead-face-identify
```
```
1. collect peoples image you wanna identify
2. put it in `images` directory, subdirectory named by peoples' name
3. put video in root directory
4. change video name in `run_demo.py` to keep consistent
```
```
>>> python crop_face.py
>>> python vgg_face_feature.py
>>> python run_demo.py
```
Details refer to my repo. If you have better results, let me know.

### **Reference**
- [MTCNN](https://arxiv.org/abs/1604.02878) for face detection
- [VGG_face2](https://arxiv.org/abs/1710.08092) for face identify
- [Chengwei's blog](https://www.dlology.com/blog/live-face-identification-with-pre-trained-vggface2-model/)


That's all, hope this help you. :)



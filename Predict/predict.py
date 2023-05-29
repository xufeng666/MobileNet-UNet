import cv2
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from images_prepare import *
import numpy as np
from tensorflow.keras.models import load_model
path = "images/"
X,X_sizes=pre_images_predict((512,512),path,True)
X=np.float32(X/255)
new_model = tf.keras.models.load_model('../content2/dental_xray_seg_150_1.h5',custom_objects={'relu6': keras.layers.ReLU(6.)})
preds = new_model.predict(X)
predict=preds[0,:,:,0]
plt.imshow(predict)
plt.imsave("1032predict150_1.png",predict)

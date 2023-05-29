from download_dataset import *
from keras.callbacks import TensorBoard
import os
path = "content2/Data"
if os.path.exists(path+'/DentalPanoramicXrays.zip') == False:
  os.mkdir(path)
  download_dataset(path+'/')

from images_prepare import *
#pre_images(resize_shape,path,include_zip)
X,X_sizes=pre_images((512,512),path,True)

from masks_prepare import *
#Y=pre_masks(resize_shape=(512,512),path='/content/Segmentation-of-Teeth-in-Panoramic-X-ray-Image/Original_Masks')  ORIGINALL MASKS function
Y=pre_splitted_masks(path='Custom_Masks') #Custom Splitted MASKS size 512x512

X=np.float32(X/255)
Y=np.float32(Y/255)
x_train=X[:105,:,:,:]
y_train=Y[:105,:,:,:]
x_test=X[105:,:,:,:]
y_test=Y[105:,:,:,:]

import cv2

import albumentations as A
#Augmention . Change what you want ! Care about Your GPU and CPU RAM

#If you get error : cannot import name '_registerMatType' from 'cv2.cv2' :
#!pip uninstall opencv-python-headless==4.5.5.62
#!pip install opencv-python-headless==4.5.2.52

aug = A.Compose([
    A.OneOf([A.RandomCrop(width=512, height=512),
                 A.PadIfNeeded(min_height=512, min_width=512, p=0.5)],p=0.4),
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25,p=0.5),
    A.Compose([A.RandomScale(scale_limit=(-0.15, 0.15), p=1, interpolation=1),
                            A.PadIfNeeded(512, 512, border_mode=cv2.BORDER_CONSTANT),
                            A.Resize(512, 512, cv2.INTER_NEAREST), ],p=0.5),
    A.ShiftScaleRotate (shift_limit=0.325, scale_limit=0.15, rotate_limit=15,border_mode=cv2.BORDER_CONSTANT, p=1),
    A.Rotate(15,p=0.5),
    A.Blur(blur_limit=1, p=0.5),
    A.Downscale(scale_min=0.15, scale_max=0.25,  always_apply=False, p=0.5),
    A.GaussNoise(var_limit=(0.05, 0.1), mean=0, per_channel=True, always_apply=False, p=0.5),
    A.HorizontalFlip(p=0.25),
])

x_train1=np.copy(x_train)
y_train1=np.copy(y_train)
count=0
while(count<4):
  x_aug2=np.copy(x_train1)
  y_aug2=np.copy(y_train1)
  for i in range(len(x_train1)):
    augmented=aug(image=x_train1[i,:,:,:],mask=y_train1[i,:,:,:])
    x_aug2[i,:,:,:]= augmented['image']
    y_aug2[i,:,:,:]= augmented['mask']
  x_train=np.concatenate((x_train,x_aug2))
  y_train=np.concatenate((y_train,y_aug2))
  if count == 9:
    break
  count += 1

del x_aug2
del X
del Y
del y_aug2
del y_train1
del x_train1
del augmented

import random
import matplotlib.pyplot as plt
random_number=random.randint(0,104)
print(random_number)

#Checking data X  and Y matching
plt.imshow(x_train[random_number,:,:,0])

#%%

#Checking data X  and Y matching
plt.imshow(y_train[random_number,:,:,0])

from model import *
from mymodel import *
from GAMModel import Mobilev2_UNET
model=Mobilev2_UNET(input_shape=(512,512,1),last_activation='sigmoid')
model.summary()



model.compile(optimizer ='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
Tensorboard= TensorBoard(log_dir="model", histogram_freq=1,write_grads=True)
history=model.fit(x_train, y_train, batch_size=2, epochs=300,verbose=1, callbacks=[Tensorboard],validation_split=0.1)



predict_img=model.predict(x_test)
##model.save(path)
predict=predict_img[1,:,:,0]

#Example Test
from sklearn.metrics import f1_score
import numpy as np
predict_img1=(predict_img>0.25)*1
y_test1=(y_test>0.25)*1

print(f1_score(predict_img1.flatten(), y_test1.flatten(), average='micro'))

plt.figure(figsize = (20,10))
plt.title("Predict Mask",fontsize = 40)
plt.imshow(predict)
#For CCA, we saved
plt.imsave("content/predict2.png",predict)

# from google.colab.patches import cv2_imshow
# import cv2
# from CCA_Analysis import *
#
#
# ##Plotting - RESULT Example
# img=cv2.imread("/content/Data/Images/107.png")#original img 107.png
#
# predict1 = cv2.resize(predict, (img.shape[1],img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
#
# mask=np.uint8(predict1*255)#
# _, mask = cv2.threshold(mask, thresh=255/2, maxval=255, type=cv2.THRESH_BINARY)
# cnts,hieararch=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# img = cv2.drawContours(img[:,:,0], cnts, -1, (255, 0, 0) , 2)
# img = cv2.UMat.get(img)
# cv2_imshow(img)

#%%

# from google.colab.patches import cv2_imshow
# import cv2
# from CCA_Analysis import *
#
#
# ##Plotting - RESULT Example with CCA_Analysis
# img=cv2.imread("/content/Data/Images/107.png")#original img 107.png
#
# #load image (mask was saved by matplotlib.pyplot)
# predicted=cv2.imread("/content/predict2.png")
#
# predicted = cv2.resize(predicted, (img.shape[1],img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
#
# cca_result,teeth_count=CCA_Analysis(img,predicted,3,2)
# cv2_imshow(cca_result)

#%%

# print(teeth_count,"Teeth Count")

#%%

import tensorflow as tf
tf.keras.models.save_model(model, 'content2/dental_xray_seg_300_2.h5')
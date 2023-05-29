import cv2
import matplotlib.pyplot as plt
image = cv2.imread('predict2.png')
image2 = cv2.imread('images/1023.jpg')
x = image2.shape[0]
y = image2.shape[1]
image = cv2.resize(image,(y,x))

plt.imshow(image)
#For CCA, we saved
plt.imsave("predict2.png",image)
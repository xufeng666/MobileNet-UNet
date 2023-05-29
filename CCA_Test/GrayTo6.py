# 对数变换
import cv2
import numpy as np
from matplotlib import pyplot as plt

#对数变换
def log(c,DA_img):
    new_img=c*np.log(1.0+DA_img)
    new_img=np.uint8(new_img+0.5)
    return new_img

img=cv2.imread("images/1024.jpg")
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
new_img=log(30,gray_img)


cv2.imshow("Gray Image", new_img)

#等待显示
cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.rcParams['font.sans-serif']=['SimHei']#解决中文乱码问题
#
# img1=plt.subplot(1,2,1)
# img1.set_title("原始图像")
# plt.imshow(gray_img,cmap="gray")
# plt.xticks([])
# plt.yticks([])
#
# img2=plt.subplot(1,2,2)
# img2.set_title("对数变换")
# plt.imshow(new_img,cmap="gray")
# plt.xticks([])
# plt.yticks([])
#
# plt.show()
# cv2.waitKey(0)

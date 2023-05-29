# Gamma变换--非线性变换
import cv2
from skimage import exposure
from matplotlib import pyplot as plt

img=cv2.imread("images/1024.jpg")
grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gamma_img=exposure.adjust_gamma(grayImg,1.8)


cv2.imshow("Gray Image", gamma_img)

#等待显示
cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.rcParams['font.sans-serif']=['SimHei']#解决中文乱码问题
# img1=plt.subplot(1,2,1)
# img1.set_title("原始图像")
# plt.imshow(grayImg,cmap="gray")
# plt.xticks([])
# plt.yticks([])
#
# img2=plt.subplot(1,2,2)
# img2.set_title("Gamma变换")
# plt.imshow(gamma_img,cmap="gray")
# plt.xticks([])
# plt.yticks([])
#
# plt.show()
# cv2.waitKey(0)


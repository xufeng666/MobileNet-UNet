import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage import exposure

grayImage=cv2.imread('images/1024.jpg') #读取图片，若图片就是灰度图就不用第二行
grayImage = cv2.cvtColor(grayImage, cv2.COLOR_BGR2GRAY) #将BGR转为灰度图


fig,ax1= plt.subplots(1, 1)
ax1.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))


hist = cv2.calcHist([grayImage],[0],None,[256],[0,256])
plt.plot(hist,color="gray")
ax1.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.xlim([0,256])
ax1.set_xlabel("range of pixel values")
ax1.set_ylabel("nums of pixel")
ax1.legend(["Gray"],loc="upper right")

plt.savefig(r'numberOfpixels6.png',dpi=300,format='svg')
plt.show()
print("finish!")

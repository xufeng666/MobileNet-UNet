from CCA_Analysis import *



img=cv2.imread("images/1028.jpg")
predict=cv2.imread("images/predict3.png")
predict=cv2.cvtColor(predict,cv2.COLOR_BGR2GRAY)
predict1 = cv2.resize(predict, (img.shape[1],img.shape[0]), interpolation=cv2.INTER_LANCZOS4)

cv2.imwrite('cca0101.png',predict1)

mask=np.uint8(predict1*255)
_, mask = cv2.threshold(mask, thresh=255/2, maxval=255, type=cv2.THRESH_BINARY)
cnts,hieararch=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
output=img[:,:,0].copy()
img = cv2.drawContours(output, cnts, -1, (255, 0, 0) , 2)

cv2.imwrite('cca0102.png',img)
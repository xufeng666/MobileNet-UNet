
from CCA_Analysis import *



img=cv2.imread("images/1028.jpg")

predicted=cv2.imread("images/predict3.png")

predicted = cv2.resize(predicted, (img.shape[1],img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
cv2.imwrite('cca0202.png',predicted)
cca_result,teeth_count=CCA_Analysis(img,predicted,1,1)
cv2.imwrite('cca0201.png',img)
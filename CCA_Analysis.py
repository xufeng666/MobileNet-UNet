import cv2
import numpy as np
from imutils import perspective
from scipy.spatial import distance as dist
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
# Load in image, convert to gray scale, and Otsu's threshold

#Function accept cv2 type
#Only useable splitted masks 

def CCA_Analysis(orig_image,predict_image,erode_iteration,open_iteration):
    kernel1 =( np.ones((5,5), dtype=np.float32))
    kernel_sharpening = np.array([[-1,-1,-1],
                                  [-1,9,-1],
                                 [-1,-1,-1]])
    image = predict_image
    image2 =orig_image    
    image=cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel1,iterations=open_iteration )
    print(kernel1)
    cv2.imwrite('cca01.png', image)
    image = cv2.filter2D(image, -1, kernel_sharpening)
    cv2.imwrite('cca02.png', image)
    image=cv2.erode(image,kernel1,iterations =erode_iteration)
    cv2.imwrite('cca03.png', image)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('cca04.png', image)
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.imwrite('cca05.png', thresh)
    labels=cv2.connectedComponents(thresh,connectivity=8)[1]
    cv2.imwrite('cca06.png', labels)
    a=np.unique(labels)

    count2=0
    for label in a:
        if label == 0:
            continue
    
        # 创建 mask
        mask = np.zeros(thresh.shape, dtype="uint8")
        mask[labels == label] = 255
        # 找到轮廓并确定轮廓区域
        cnts,hieararch = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0]
        c_area = cv2.contourArea(cnts)
        # threshhold for tooth count
        if c_area>2000:
            count2+=1
        
        (x,y),radius = cv2.minEnclosingCircle(cnts)
        rect = cv2.minAreaRect(cnts)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="int")    
        box = perspective.order_points(box)
        color1 = (list(np.random.choice(range(150), size=3)))  
        color =[int(color1[0]), int(color1[1]), int(color1[2])]  
        cv2.drawContours(image2,[box.astype("int")],0,color,1)
        (tl,tr,br,bl)=box

        (tltrX,tltrY)=midpoint(tl,tr)
        (blbrX,blbrY)=midpoint(bl,br)
    	# 计算之间的中点左上方和右上方的点;
    	# 紧随其后的是右上角和右下角之间的中点;
        (tlblX,tlblY)=midpoint(tl,bl)
        (trbrX,trbrY)=midpoint(tr,br)
    	# 画图像中点
        cv2.circle(image2, (int(tltrX), int(tltrY)), 2, (255, 0, 0), -1)
        cv2.circle(image2, (int(blbrX), int(blbrY)), 2, (255, 0, 0), -1)
        cv2.circle(image2, (int(tlblX), int(tlblY)), 2, (255, 0, 0), -1)
        cv2.circle(image2, (int(trbrX), int(trbrY)), 2, (255, 0, 0), -1)
        cv2.line(image2, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),color, 1)
        cv2.line(image2, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),color, 1)
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        
        
        
        pixelsPerMetric=1
        dimA = dA * pixelsPerMetric
        dimB = dB *pixelsPerMetric
        # cv2.putText(image2, "{:.1f}pixel".format(dimA),(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,0.25, color, 1)
        # cv2.putText(image2, "{:.1f}pixel".format(dimB),(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,0.25, color, 1)
        cv2.putText(image2, "{:.0f}".format(label),(int(tltrX - 5), int(tltrY - 5)), cv2.FONT_HERSHEY_SIMPLEX,0.35, color, 1)
    teeth_count=count2
    return image2,teeth_count






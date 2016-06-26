import cv2
img1 = cv2.imread('smartTV.jpg')
img2 = cv2.imread('smart.jpg')

dst = cv2.addWeighted(img1,0.5,img2,0.1,0)

cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

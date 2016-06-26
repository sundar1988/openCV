import cv2

#Load original image
image=cv2.imread('smartTV.jpg')
print image
cv2.imshow('this is original image', image)

cv2.waitKey(0)
#Show grayscale image
image=cv2.imread('smartTV.jpg', cv2.IMREAD_GRAYSCALE)
print image
cv2.imshow('this is grayscale image', image)

if image is None:
    print ('No image exist')

else:
    cv2.waitKey(0)
    cv2.destroyAllWindows()

import cv2


img = cv2.imread('x.jpg')
dst = cv2.medianBlur(img, 3)

img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])


img_fix = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

cv2.imwrite('save.jpg', dst)
cv2.imwrite('fix.jpg', img_fix)
cv2.waitKey(0)
cv2.destroyAllWindows()
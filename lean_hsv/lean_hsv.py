import cv2
import numpy as np
def noting(x):
    # 트랙바가 변경될 때마다 호출되는 콜백 함수
    l_h = cv2.getTrackbarPos("L - H", "trackbers")
    l_s = cv2.getTrackbarPos("L - S", "trackbers")
    l_v = cv2.getTrackbarPos("L - V", "trackbers")
    u_h = cv2.getTrackbarPos("U - H", "trackbers")
    u_s = cv2.getTrackbarPos("U - S", "trackbers")
    u_v = cv2.getTrackbarPos("U - V", "trackbers")

    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])
    np.save("C:/Users/lim/Desktop/data/hsv", lower, upper)
    img = cv2.imread("C:/Users/lim/Desktop/data/KakaoTalk_20240304_213402136.png")
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, lower, upper)
    cv2.imshow("lean", mask)

cv2.namedWindow("trackbers")
cv2.createTrackbar("L - H", "trackbers", 0, 255, noting)
cv2.createTrackbar("L - S", "trackbers", 0, 255, noting)
cv2.createTrackbar("L - V", "trackbers", 200, 255, noting)
cv2.createTrackbar("U - H", "trackbers", 255, 255, noting)
cv2.createTrackbar("U - S", "trackbers", 50, 255, noting)
cv2.createTrackbar("U - V", "trackbers", 255, 255, noting)

cv2.waitKey()
cv2.destroyAllWindows()

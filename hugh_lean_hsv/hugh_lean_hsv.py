import cv2
import numpy as np
def draw_line(img, lines, color=[0,255,0], thickness=3):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1 ,y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_line(line_img, lines)
    return line_img

def weighted_img(lines, img, alpha=0.8, beta=1):
    return cv2.addWeighted(lines, alpha, img, beta, 0)
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,)*channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
def mask_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv_lower = np.array([10, 0, 150])
    hsv_upper = np.array([255, 255, 255])
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    return mask
def canny_img(img):
    canny_image = cv2.Canny(img, 120,200)
    return canny_image

def draw_line(img, lines, color=[0,255,0], thickness=3):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1 ,y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_line(line_img, lines)
    return line_img

def weighted_img(lines, img, alpha=0.8, beta=1):
    return cv2.addWeighted(lines, alpha, img, beta, 0)

img = cv2.imread("C:/Users/lim/Desktop/data/KakaoTalk_20240304_213402136.png")
mask = mask_hsv(img)
canny = canny_img(mask)
imshape = img.shape  # ROI region
vertices = np.array([[(0, imshape[0]), (70,81), (140, 81), (imshape[1], imshape[0])]], dtype=np.int32)
roi = region_of_interest (canny, vertices)
lines = hough_lines(roi, 2, np.pi / 180, 50, 50, 200)
lines_edges = weighted_img(lines, img, alpha=0.8, beta=1.)

cv2.imshow("original", img)
cv2.imshow("mask", mask)
cv2.imshow("canny", canny)
cv2.imshow("roi", roi)
cv2.imshow("lean", lines_edges)
cv2.waitKey()
cv2.destroyAllWindows()
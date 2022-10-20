import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils

def tinh_hist(img):
    hist = np.zeros((256,), np.uint8)  # tạo mảng để thống kê số lượng pixel cho từng mức sáng
    h, w = img.shape[:2]  # lấy số đo của ảnh
    for i in range(h):
        for j in range(w):
            hist[img[i][j]] += 1
    return hist

def can_bang_hist(hist):
    trans = np.zeros_like(hist, np.float64)
    for i in range(len(trans)):
        trans[i] = hist[:i].sum()
    new_hist = (trans - trans.min()) / (trans.max() - trans.min()) * 255
    new_hist = np.uint8(new_hist)
    return new_hist

def contrast_stretching(img):
    hist = tinh_hist(img).ravel()
    new_hist = can_bang_hist(hist)

    h, w = img.shape[:2]
    for i in range(h):
        for j in range(w):
            img[i, j] = new_hist[img[i, j]]
    
    return img



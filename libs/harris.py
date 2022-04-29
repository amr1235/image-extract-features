from scipy import signal as sig
import numpy as np
from scipy import ndimage as ndi
import cv2
def apply_harris(img, k = 0.05, threshold = 0.01):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    I_x = sig.convolve2d(img_gray, kernel_x, mode='same')
    I_y = sig.convolve2d(img_gray, kernel_y, mode='same')
    Ixx = ndi.gaussian_filter(I_x**2, sigma=1)
    Ixy = ndi.gaussian_filter(I_y*I_x, sigma=1)
    Iyy = ndi.gaussian_filter(I_y**2, sigma=1)

    # determinant
    detM = Ixx * Iyy - Ixy ** 2
    # trace
    traceM = Ixx + Iyy
        
    harrisR = detM - k * traceM ** 2
    img_copy = np.copy(img)
    img_copy[np.where(harrisR>threshold*harrisR.max())] = [255,0,255]
   
    return img_copy
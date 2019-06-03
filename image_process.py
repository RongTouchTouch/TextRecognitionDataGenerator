import numpy as np
import cv2
import random
import copy


def add_noise(img, noise):
    for i in range(noise):  # 添加点噪声
        temp_x = np.random.randint(0, img.shape[0])
        temp_y = np.random.randint(0, img.shape[1])
        img[temp_x][temp_y] = 255
    return img


def add_erode(img,erode):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erode, erode))
    img = cv2.erode(img, kernel)
    return img


def add_dilate(img,dilate):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate, dilate))
    img = cv2.dilate(img, kernel)
    return img

def add_incline(img, incline):
    temp = img
    pts1 = np.float32([[0, 0],[img.shape[0], 0],[0, img.shape[1]],[img.shape[0], img.shape[1]]])
    pts2 = np.float32([[incline, 0],[img.shape[0]+incline, 0],[0, img.shape[1]],[img.shape[0], img.shape[1]]])
    warp_mat = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, warp_mat, (img.shape[1]+incline, img.shape[0]),borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
    return img[0:img.shape[0], incline:img.shape[1]]


def do(random_process, noise, dilate, erode,incline, img):
    im = copy.deepcopy(img)
    if random_process:
        if noise and random.random() < 0.5:
            im = add_noise(im,noise)
        else:
            noise = 0
        if dilate and random.random() < 0.25:
            im = add_dilate(im,dilate)
        else:
            dilate = 0
        if erode and random.random() < 0.25:
            im = add_erode(im,erode)
        else:
            erode = 0
        if incline and random.random() < 0.25:
            im = add_incline(im,incline)
        else:
            incline = 0
    else:
        if noise:
            im = add_noise(im,noise)
        if dilate:
            im = add_dilate(im,dilate)
        if erode:
            im = add_erode(im,erode)
        if incline:
            im = add_incline(im,incline)
    return im, noise, dilate, erode, incline

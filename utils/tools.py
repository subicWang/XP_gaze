# -*- coding: utf-8 -*-
"""
Aouther: Subic
Time: 2019/3/25: 11:21
"""
import numpy as np
import cv2


def expand(eye, width, height):
    scale = True
    min_x = eye[:, 0].min()
    min_y = eye[:, 1].min()
    max_x = eye[:, 0].max()
    max_y = eye[:, 1].max()
    k = 1.5
    w = max_x - min_x
    h = max_y - min_y
    c_x = (max_x + min_x)/2
    c_y = (max_y + min_y)/2
    expand_x = 0
    expand_y = 0
    if w / (h+1e-4) > k:
        expand_y = (w / k - h) / 2
    else:
        expand_x = (k * h - w) / 2
    if w < 18:
        expand_x = 3
        expand_y = 2
        scale = False

    min_x_new = c_x - w/2 - expand_x
    min_y_new = c_y - h/2 - expand_y
    max_x_new = c_x + w/2 + expand_x
    max_y_new = c_y + h/2 + expand_y

    eye_ori = [min_x_new, min_y_new, max_x_new, max_y_new]

    new_w = max_x_new - min_x_new
    new_h = max_y_new - min_y_new
    new_cx = (max_x_new + min_x_new)/2
    new_cy = (max_y_new + min_y_new)/2
    min_x_new = new_cx - new_w*1.5
    min_y_new = new_cy - new_h*1.5
    max_x_new = new_cx + new_w*1.5
    max_y_new = new_cy + new_h*1.5

    if min_x_new < 0:
        min_x_new = 0
    if min_y_new < 0:
        min_y_new = 0
    if max_x_new > width:
        max_x_new = width
    if max_y_new > height:
        max_y_new = height

    eye_expand = [min_x_new, min_y_new, max_x_new, max_y_new]
    return eye_ori, eye_expand, scale


def get_eye_patch(img, eye_box_ori):
    height, width = img.shape[:2]
    eye_ori, eye_new, scale = expand(eye_box_ori, width, height)

    eye_img = img[int(eye_new[1]): int(eye_new[3]), int(eye_new[0]): int(eye_new[2])]
    eye_bbox = [eye_ori[0] - eye_new[0], eye_ori[1] - eye_new[1], eye_ori[2] - eye_new[0], eye_ori[3] - eye_new[1]]
    return eye_img, eye_bbox


def adjust_gamma(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    new_img = cv2.LUT(np.array(img, dtype=np.uint8), table)
    return new_img
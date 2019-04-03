# -*- coding: utf-8 -*-
"""
Aouther: Subic
Time: 2019/2/26: 13:10
"""
import numpy as np


class RandomCrop(object):
    def __init__(self, phase="train"):
        self.phase = phase

    def __call__(self, img, eyebox):
        height, width, _ = img.shape
        w = eyebox[2] - eyebox[0]
        h = eyebox[3] - eyebox[1]
        center_x = (eyebox[2] + eyebox[0])/2
        center_y = (eyebox[3] + eyebox[1])/2
        off_x = 0
        off_y = 0
        Ratio = 0.3
        if self.phase == "train":
            expand_ratio = np.random.rand() * Ratio
            expand_w = expand_ratio * w
            expand_h = expand_ratio * h
            off_x = (-1 + 2 * np.random.rand())*0.1 * w
            off_y = (-1 + 2 * np.random.rand())*0.1 * h
        else:
            expand_w = Ratio/2 * w
            expand_h = Ratio/2 * h
        new_x0 = center_x + off_x - w/2 - expand_w
        new_y0 = center_y + off_y - h/2 - expand_h
        new_x1 = center_x + off_x + w/2 + expand_w
        new_y1 = center_y + off_y + h/2 + expand_h
        if new_x0 < 0:
            new_x0 = 0
        if new_y0 < 0:
            new_y0 = 0
        if new_x1 > width:
            new_x1 = width
        if new_y1 > height:
            new_y1 = height
        if int(new_y1)-int(new_y0) == 0 or int(new_x1) - int(new_x0)==0:
            print("error!")
        eye = img[int(new_y0): int(new_y1), int(new_x0): int(new_x1)]
        return eye

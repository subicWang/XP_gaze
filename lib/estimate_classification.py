# -*- coding: utf-8 -*-
"""
Aouther: Subic
Time: 2019/2/26: 17:21
glass: 不戴眼镜：0， 戴眼镜： 1， 戴墨镜： 2
yaw: 根据yaw判断侧脸角度。[~-50, -50~-40, -40~-30, -30~-20, -20~-10, -10~0, 0~10, 10~20, 20~30, 30~40, 40~50, 50~]共十二个等级: [-6,-5,-4,-3,-2,-1,1,2,3,4,5,6]
pitch: 根据pitch判断抬头低头角度。[~-50, -50~-40, -40~-30, -30~-20, -20~-10, -10~0, 0~10, 10~20, 20~30, 30~40, 40~50, 50~]共十二个等级: [-6,-5,-4,-3,-2,-1,1,2,3,4,5,6]
gaze: 对不同gaze角度测评。
"""
import numpy as np
import torch


class EstimateClassification(object):
    def __init__(self):
        pass
    def __call__(self, gaze, angle_error):
        K = int(60 / 10)
        Row_Num = 3 + 2 * K * 4
        Estimate = np.zeros((Row_Num, 2))
        Estimate_index = np.zeros((Row_Num, 2))
        for b in range(len(gaze)):
        # 分阶段， 每10度为一个阶段
            label = int(gaze[b, 2])
            glass = int(gaze[b, 3])
            if label > 0:
                Estimate[glass] += angle_error[b]
                Estimate_index[glass] += 1
                ii = 3
                for i in [0, 1, 4, 5]:
                    object = gaze[b, i]
                    index = int(np.floor(object/10))
                    if index > 5:
                        index = 5
                    if index < -6:
                        index = -6
                    index += 6
                    Estimate[ii + index] += angle_error[b]
                    Estimate_index[ii + index] += 1
                    ii += 2*K
        return Estimate, Estimate_index


class EstimateClassification_new(object):
    def __init__(self):
        pass
    def __call__(self, gaze, angle_error):
        K = int(60 / 10)
        Row_Num = 1 + 2 * K * 4
        Estimate = np.zeros((Row_Num, 3, 2))
        Estimate_index = np.zeros((Row_Num, 3, 2))
        for b in range(len(gaze)):
        # 分阶段， 每10度为一个阶段
            label = int(gaze[b, 2])
            glass = int(gaze[b, 2])
            if label > 0:
                Estimate[glass] += angle_error[b]
                Estimate_index[glass] += 1
                ii = 3
                for i in [0, 1, 4, 5]:
                    object = gaze[b, i]
                    index = int(np.floor(object/10))
                    if index > 5:
                        index = 5
                    if index < -6:
                        index = -6
                    index += 6
                    Estimate[ii + index] += angle_error[b]
                    Estimate_index[ii + index] += 1
                    ii += 2*K
        return Estimate, Estimate_index
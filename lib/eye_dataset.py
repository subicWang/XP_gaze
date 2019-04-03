# -*- coding: utf-8 -*-
"""
Aouther: Subic
Time: 2019/3/25: 11:45
"""
import cv2
import os
from torch.utils.data import Dataset
import torch
import pickle
import torch.utils.data
from lib.augmentation import RandomCrop
import numpy as np
import lmdb
from lmdb_data import datum_pb2_twoeyes
import visdom


class EyeDataset(Dataset):
    def __init__(self, data_path, phase):
        self.phase = phase
        self.datum = datum_pb2_twoeyes.Datum()
        db = lmdb.open(data_path, readonly=True)
        self.txn = db.begin()
        self.num = int(self.txn.get("num_samples".encode()))
        print("dataset size: ", self.num)

    def __getitem__(self, index):
        # visd = visdom.Visdom(env="gaze_vis", port=10011)
        value = self.txn.get('{:0>8d}'.format(index).encode())

        self.datum.ParseFromString(value)

        img = np.fromstring(self.datum.data, dtype=np.uint8).reshape(
            self.datum.width, self.datum.height, self.datum.channels)
        img_name = self.datum.name.decode()
        img = img.transpose((2, 0, 1))
        img = torch.Tensor(img)

        if self.phase == "test":
            return img, img_name
        else:
            label = self.datum.label
            return img, np.array(label)

    def __len__(self):
        return self.num

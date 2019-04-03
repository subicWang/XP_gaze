import cv2
import os
from torch.utils.data import Dataset
import torch
import pickle
import torch.utils.data
from lib.augmentation import RandomCrop
import numpy as np
import lmdb
# from lmdb_data import datum_twoeyes_pb2
from lmdb_data import datum_oneeye_pb2
import visdom

class GazeDataset(Dataset):
    def __init__(self, dataset_dir, lmdb_list, if_head=False, if_face=False):
        self.data = []
        for lmdb in lmdb_list:
            path = os.path.join(dataset_dir, lmdb)
            self.data += torch.load(path)
        self.if_head = if_head
        self.if_face = if_face

        print("len(self.data): ", len(self.data))

    def __getitem__(self, index):
        image = self.data[index]['image']
        gaze = self.data[index]['gaze']

        # image = cv2.resize(image, (224, 224))
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = torch.Tensor(image.transpose(2, 0, 1))
        assert len(gaze) == 2
        gaze = torch.Tensor(gaze)
        if self.if_head:
            headpose = self.data[index]['head']
            if len(headpose) == 3:
                headpose = headpose[:2]
            assert len(headpose) == 2
            headpose = torch.Tensor(headpose)

            if self.if_face:
                face = self.data[index]['face']
                if len(face.shape) == 2:
                    face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
                face = torch.Tensor(face.transpose(2, 0, 1))
                return image, gaze, headpose, face
            else:
                return image, gaze, headpose
        else:
            return image, gaze

    def __len__(self):
        return len(self.data)


class EyeDataset_TRAIN(Dataset):
    def __init__(self, phase="train"):#
        label = []
        imgs = []
        img_names = []
        self.phase = phase
        data_file_path1 = "/data-private/face/wx_space/label_mark/train_data.pkl"
        with open(data_file_path1, "rb") as data:
            imgs += pickle.load(data)
            label += pickle.load(data)
            img_names += pickle.load(data)
        data_file_path2 = "/data-private/face/wx_space/label_mark/train_data_1.pkl"
        with open(data_file_path2, "rb") as data:
            imgs += pickle.load(data)
            label += pickle.load(data)
            img_names += pickle.load(data)
        Num = len(imgs)
        Num_train = int(Num*0.8)
        np.random.seed(1)
        index = np.random.permutation(Num)
        imgs_shuffle = np.array(imgs)[index]
        labels_shuffle = np.array(label)[index]
        if phase == "train":
            self.imgs = imgs_shuffle[:Num_train]
            self.labels = labels_shuffle[:Num_train]
        elif phase == "valid":
            self.imgs = imgs_shuffle[Num_train:]
            self.labels = labels_shuffle[Num_train:]

    def __getitem__(self, index):
        # visd = visdom.Visdom(env="gaze_vis", port=10011)

        eye = self.imgs[index]
        eye = eye.transpose((2, 0, 1))
        # visd.image(eye)
        eye = torch.Tensor(eye)
        # if self.phase == "test":
        #     return eye, self.img_names[index]
        # else:
        label = self.labels[index]
        return eye, np.array(label)

    def __len__(self):
        return len(self.imgs)


class EyeDataset_TEST(Dataset):
    def __init__(self, imgs_path):#
        self.imgs = []
        self.img_names = []
        for img_name in os.listdir(imgs_path):
            img_path = os.path.join(imgs_path, img_name)
            img = cv2.imread(img_path)
            self.imgs.append(cv2.resize(img, (48, 32)))
            self.img_names.append(img_name)

    def __getitem__(self, index):
        # visd = visdom.Visdom(env="gaze_vis", port=10011)

        eye = self.imgs[index]
        eye = eye.transpose((2, 0, 1))
        # visd.image(eye)
        eye = torch.Tensor(eye)
        return eye, self.img_names[index]

    def __len__(self):
        return len(self.imgs)


class GazeDataset1(Dataset):
    def __init__(self, data_path, w=48, h=32, phase="train"):#
        self.imgs = []
        self.data = []
        self.img_names = []

        self.new_w, self.new_h = (w, h)
        self.randomcrop = RandomCrop(phase)
        if phase != "test":
            for data_file in os.listdir(data_path):
                data_file_path = os.path.join(data_path, data_file)
                with open(data_file_path, "rb") as data:
                    self.img_names += pickle.load(data)
                    self.imgs += pickle.load(data)
                    self.data += pickle.load(data)
        else:
            data_file = "gaze_detect_11.16.pkl"
            data_file_path = os.path.join(data_path, data_file)
            with open(data_file_path, "rb") as data:
                self.imgs += pickle.load(data)
                self.data += pickle.load(data)


    def __getitem__(self, index):
        # visd = visdom.Visdom(env="gaze_vis", port=10011)
        eyebox = self.data[index][:4]
        gaze = self.data[index][4:]
        img = self.imgs[index]
        height, width, _ = img.shape
        eye1 = self.randomcrop(img, eyebox)

        # img_draw = eye.transpose((2, 0, 1))
        # visd.image(img_draw)

        eye = cv2.resize(eye1, (self.new_w, self.new_h))
        eye = eye.transpose((2, 0, 1))
        eye = torch.FloatTensor(eye)

        # eye = self.transform(eye)

        return eye, np.array(gaze, dtype=np.float32)

    def __len__(self):
        return len(self.data)


class GazeLmdbDatasetOneEye(Dataset):
    def __init__(self, data_path, w=48, h=32, transform=None, phase="train"):#
        self.transform = transform

        self.new_w, self.new_h = (w, h)
        self.randomcrop = RandomCrop(phase)
        db = lmdb.open(data_path, readonly=True)
        # self.datum = datum_pb2_twoeyes.Datum()
        self.datum = datum_oneeye_pb2.Datum()
        self.txn = db.begin()
        self.num = int(self.txn.get('num_samples'.encode()))
        # self.num = 5000
        print('dataset size:', self.num)

    def __getitem__(self, index):
        # visd = visdom.Visdom(env="gaze_vis_dataset", port=10011)
        value = self.txn.get('{:0>8d}'.format(index).encode())
        self.datum.ParseFromString(value)
        img = np.fromstring(self.datum.data, dtype=np.uint8).reshape(
            self.datum.width, self.datum.height, self.datum.channels)

        # img_draw = img.transpose((2, 0, 1))
        # visd.image(img_draw)

        eyebox = [self.datum.box_left, self.datum.box_top, self.datum.box_right, self.datum.box_bottom]
        gaze = [self.datum.yaw, self.datum.pitch]
        label = self.datum.label

        eye1 = self.randomcrop(img, eyebox)
        # name = self.img_names[index//2]
        # img_draw = eye1.transpose((2, 0, 1))
        # visd.image(img_draw)
        eye = cv2.resize(eye1, (self.new_w, self.new_h))
        eye = eye.transpose((2, 0, 1))
        eye = torch.Tensor(eye)
        gaze = torch.Tensor(gaze)
        name = self.datum.name.decode()

        return eye, gaze, label, [name]

    def __len__(self):
        return self.num


# class GazeLmdbDatasetTwoEyes(Dataset):
#     def __init__(self, data_path, w=48, h=32, transform=None, phase="train"):#
#         self.transform = transform
#
#         self.new_w, self.new_h = (w, h)
#         self.randomcrop = RandomCrop(phase)
#         db = lmdb.open(data_path, readonly=True)
#         self.datum = datum_twoeyes_pb2.Datum()
#         self.txn = db.begin()
#         self.num = int(self.txn.get('num_samples'.encode()))
#         print('dataset size:', self.num)
#
#     def __getitem__(self, index):
#         # visd = visdom.Visdom(env="gaze_vis_dataset", port=10011)
#
#         value = self.txn.get('{:0>8d}'.format(index).encode())
#         self.datum.ParseFromString(value)
#         img_left = np.fromstring(self.datum.data_left, dtype=np.uint8).reshape(
#             self.datum.width_left, self.datum.height_left, self.datum.channels_left)
#         img_right = np.fromstring(self.datum.data_right, dtype=np.uint8).reshape(
#             self.datum.width_right, self.datum.height_right, self.datum.channels_right)
#
#         # img_draw = img.transpose((2, 0, 1))
#         # visd.image(img_draw)
#
#         eyebox_left = [self.datum.box_left_left, self.datum.box_top_left,
#                        self.datum.box_right_left, self.datum.box_bottom_left]
#         gaze_left = [self.datum.yaw_left, self.datum.pitch_left]
#         label_left = self.datum.label_left
#         eye_left = self.randomcrop(img_left, eyebox_left)
#         try:
#             eye_left = cv2.resize(eye_left, (self.new_w, self.new_h))
#         except:
#             print("left")
#         eyebox_right = [self.datum.box_left_right, self.datum.box_top_right,
#                        self.datum.box_right_right, self.datum.box_bottom_right]
#         gaze_right = [self.datum.yaw_right, self.datum.pitch_right]
#         label_right = self.datum.label_right
#         eye_right = self.randomcrop(img_right, eyebox_right)
#         try:
#             eye_right = cv2.resize(eye_right, (self.new_w, self.new_h))
#         except:
#             print("right")
#         # name = self.img_names[index//2]
#         # img_draw = eye1.transpose((2, 0, 1))
#         # visd.image(img_draw)
#
#         eye = np.concatenate((eye_left, eye_right), axis=1)
#         eye = eye.transpose((2, 0, 1))
#         eye = torch.Tensor(eye)
#         gaze = [gaze_left[0], gaze_left[1], gaze_right[0], gaze_right[1]]
#         gaze = torch.Tensor(gaze)
#         label = np.array([label_left, label_right])
#         name = self.datum.name.decode()
#         return eye, gaze, label, [name]
#
#     def __len__(self):
#         return self.num
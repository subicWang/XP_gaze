import math
import os

import cv2

from PIL import Image
import numpy as np
from tqdm import tqdm

import lmdb_data.datum_pb2_twoeyes as datum_pb2
import lmdb
from utils.tools import expand, get_eye_patch


def array_to_datum(arr, label, gaze, vector, name, bbox):
    assert arr.ndim == 3
    assert arr.dtype == np.uint8
    assert label is not None
    assert len(gaze) == 2
    assert len(vector) == 3
    assert len(bbox) == 4
    datum = datum_pb2.Datum()
    datum.width, datum.height, datum.channels = arr.shape
    datum.data = arr.tostring()
    datum.label = label
    datum.name = bytes(name, encoding='utf8')
    datum.yaw, datum.pitch = gaze
    datum.v_x, datum.v_y, datum.v_z = vector
    datum.box_left, datum.box_top, datum.box_right, datum.box_bottom = bbox
    return datum


def array_to_datum_LeftRight(data):
    img_left, label_left, gaze_left, vector_left, bbox_left, img_right, label_right, gaze_right, vector_right, bbox_right, name = data
    assert img_left.ndim == 3
    assert img_right.ndim == 3
    assert img_left.dtype == np.uint8
    assert img_right.dtype == np.uint8
    assert label_left is not None
    assert label_right is not None
    assert len(gaze_left) == 2
    assert len(gaze_right) == 2
    assert len(vector_left) == 3
    assert len(vector_right) == 3
    assert len(bbox_left) == 4
    assert len(bbox_right) == 4
    datum = datum_pb2.Datum()
    # datum.width, datum.height, datum.channels = img_left.shape
    datum.width_left, datum.height_left, datum.channels_left = img_left.shape
    datum.width_right, datum.height_right, datum.channels_right = img_right.shape
    datum.data_left = img_left.tostring()
    datum.data_right = img_right.tostring()
    datum.label_left = label_left
    datum.label_right = label_right
    datum.name = bytes(name, encoding='utf8')
    datum.yaw_left, datum.pitch_left = gaze_left
    datum.yaw_right, datum.pitch_right = gaze_right
    datum.v_x_left, datum.v_y_left, datum.v_z_left = vector_left
    datum.v_x_right, datum.v_y_right, datum.v_z_right = vector_right
    datum.box_left_left, datum.box_top_left, datum.box_right_left, datum.box_bottom_left = bbox_left
    datum.box_left_right, datum.box_top_right, datum.box_right_right, datum.box_bottom_right = bbox_right
    return datum
def preprocess(img):
    # TODO put your code here
    return np.asarray(img, dtype=np.uint8)


def save_to_lmdb(save_path, vector_gt_paths):
    """
    :param save_path: lmdb_data path(dir, not file)
    :param vector_gt_path: img path and vector list path
    """
    lines = []
    for vector_gt_path in vector_gt_paths:
        with open(vector_gt_path, 'r') as f:
            lines += f.readlines()
    predict_labels = "/data-private/face/wx_space/label_mark/gaze_label_mark/predict_label/"
    label_dict = {}
    for label_path in os.listdir(predict_labels):
        with open(os.path.join(predict_labels, label_path), "r") as f:
            for line in f.readlines():
                line = line.strip().split("\t")

                name = label_path.replace(".txt", "")+"/"+line[0].split("_")[1]+"/"+line[0]
                label_dict[name] = int(line[1])

    db = lmdb.open(save_path, map_size=1024 ** 4)
    txn = db.begin(write=True)

    count = 0
    for line in tqdm(lines):
        split = line.split()
        img_path = split[0]
        img = cv2.imread("/data-private/face/xp_data/" + img_path)
        points = np.array(list(map(float, split[1:])))
        landmarks = points[:-6].reshape(-1, 2)
        eyebox_left_ori = landmarks[[2, 3, 8, 11, 12], :]
        eyebox_right_ori = landmarks[[0, 1, 7, 9, 10], :]
        left_eye_img, left_eye_box = get_eye_patch(img, eyebox_left_ori)
        right_eye_img, right_eye_box = get_eye_patch(img, eyebox_right_ori)

        left_vector = points[-6:-3]
        right_vector = points[-3:]
        img_left_path = img_path.replace(".png", "_left.jpg")
        img_right_path = img_path.replace(".png", "_right.jpg")
        left_eye_label = label_dict[img_left_path]
        right_eye_label = label_dict[img_right_path]

        left_gaze = [-math.asin(left_vector[0]) * 180 / math.pi, -math.asin(left_vector[1]) * 180 / math.pi]
        right_gaze = [-math.asin(right_vector[0]) * 180 / math.pi, -math.asin(right_vector[1]) * 180 / math.pi]

        datum_left = array_to_datum(left_eye_img, left_eye_label, left_gaze, left_vector, img_left_path, left_eye_box)

        txn.put('{:0>8d}'.format(count).encode(), datum_left.SerializeToString())
        count += 1
        datum_right = array_to_datum(right_eye_img, right_eye_label, right_gaze, right_vector, img_right_path, right_eye_box)
        txn.put('{:0>8d}'.format(count).encode(), datum_right.SerializeToString())
        count += 1
        # if count % 1000 == 0:
        #     print('processed %d images' % count)

    print('num_samples: ', count)
    txn.put('num_samples'.encode(), str(count).encode())
    txn.commit()
    db.close()

def save_to_lmdb_LeftRight(save_path, vector_gt_paths):
    """
    :param save_path: lmdb_data path(dir, not file)
    :param vector_gt_path: img path and vector list path
    """
    lines = []
    for vector_gt_path in vector_gt_paths:
        with open(vector_gt_path, 'r') as f:
            lines += f.readlines()
    predict_labels = "/data-private/face/wx_space/label_mark/gaze_label_mark/predict_label/"
    label_dict = {}
    for label_path in os.listdir(predict_labels):
        with open(os.path.join(predict_labels, label_path), "r") as f:
            for line in f.readlines():
                line = line.strip().split("\t")

                name = label_path.replace(".txt", "")+"/"+line[0].split("_")[1]+"/"+line[0]
                label_dict[name] = int(line[1])

    db = lmdb.open(save_path, map_size=1024 ** 4)
    txn = db.begin(write=True)

    count = 0
    for line in tqdm(lines):
        split = line.split()
        img_path = split[0]
        img = cv2.imread("/data-private/face/xp_data/" + img_path)
        points = np.array(list(map(float, split[1:])))
        landmarks = points[:-6].reshape(-1, 2)
        eyebox_left_ori = landmarks[[2, 3, 8, 11, 12], :]
        eyebox_right_ori = landmarks[[0, 1, 7, 9, 10], :]
        left_eye_img, left_eye_box = get_eye_patch(img, eyebox_left_ori)
        right_eye_img, right_eye_box = get_eye_patch(img, eyebox_right_ori)

        left_vector = points[-6:-3]
        right_vector = points[-3:]

        left_eye_label = label_dict[img_path.replace(".png", "_left.jpg")]
        right_eye_label = label_dict[img_path.replace(".png", "_right.jpg")]

        left_gaze = [-math.asin(left_vector[0]) * 180 / math.pi, -math.asin(left_vector[1]) * 180 / math.pi]
        right_gaze = [-math.asin(right_vector[0]) * 180 / math.pi, -math.asin(right_vector[1]) * 180 / math.pi]
        data = [left_eye_img, left_eye_label, left_gaze, left_vector, left_eye_box, right_eye_img,
                right_eye_label, right_gaze, right_vector, right_eye_box, img_path]

        datum = array_to_datum_LeftRight(data)
        txn.put('{:0>8d}'.format(count).encode(), datum.SerializeToString())
        count += 1

    print('num_samples: ', count)
    txn.put('num_samples'.encode(), str(count).encode())
    txn.commit()
    db.close()


if '__main__' == __name__:
    save_dir = '/data-private/face/gaze/lmdbOneEye'
    train_paths = [
        '/data-private/face/gaze/gt_file/vector/gaze_gt_11.07_vector_0227.txt',
        '/data-private/face/gaze/gt_file/vector/gaze_gt_11.08_vector_0227.txt',
        '/data-private/face/gaze/gt_file/vector/gaze_gt_11.09_vector_0227.txt',
        '/data-private/face/gaze/gt_file/vector/gaze_gt_11.10_vector_0227.txt',
        '/data-private/face/gaze/gt_file/vector/gaze_gt_11.12_vector_0227.txt',
        '/data-private/face/gaze/gt_file/vector/gaze_gt_11.13_vector_0227.txt',
        '/data-private/face/gaze/gt_file/vector/gaze_gt_11.14_vector_0227.txt',
        '/data-private/face/gaze/gt_file/vector/gaze_gt_11.15_vector_0227.txt',
        '/data-private/face/gaze/gt_file/vector/gaze_gt_11.17_vector_0227.txt',
        '/data-private/face/gaze/gt_file/vector/gaze_gt_11.19_vector_0227.txt',
        '/data-private/face/gaze/gt_file/vector/gaze_gt_11.25_vector_0227.txt',
        '/data-private/face/gaze/gt_file/vector/gaze_gt_11.26_vector_0227.txt',
        '/data-private/face/gaze/gt_file/vector/gaze_gt_11.27_vector_0227.txt',
        '/data-private/face/gaze/gt_file/vector/gaze_gt_11.28_vector_0227.txt',
        '/data-private/face/gaze/gt_file/vector/gaze_gt_11.29_vector_0227.txt',
        '/data-private/face/gaze/gt_file/vector/gaze_gt_11.30_vector_0227.txt',
        '/data-private/face/gaze/gt_file/vector/gaze_gt_12.01_vector_0227.txt',
        '/data-private/face/gaze/gt_file/vector/gaze_gt_12.02_vector_0227.txt',
        '/data-private/face/gaze/gt_file/vector/gaze_gt_12.03_vector_0227.txt',
        '/data-private/face/gaze/gt_file/vector/gaze_gt_12.05_vector_0227.txt',
        '/data-private/face/gaze/gt_file/vector/gaze_gt_12.06_vector_0227.txt',
    ]
    val_paths = [
        '/data-private/face/gaze/gt_file/vector/gaze_gt_12.04_vector_0227.txt',
    ]
    test_paths = [
        '/data-private/face/gaze/gt_file/vector/gaze_gt_11.16_vector_0227.txt',
    ]
    save_to_lmdb(os.path.join(save_dir, 'train.db'), train_paths)
    save_to_lmdb(os.path.join(save_dir, 'val.db'), val_paths)
    save_to_lmdb(os.path.join(save_dir, 'test.db'), test_paths)

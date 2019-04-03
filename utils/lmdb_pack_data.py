# -*- coding: utf-8 -*-
"""
Aouther: Subic
Time: 2019/3/25: 19:02
"""
import os
import cv2
import lmdb
import numpy as np
import lmdb_data.datum_pb2_twoeyes as datum_pb2


def array_to_datum_train(name, img, label):
    assert img.ndim == 3
    assert img.dtype == np.uint8
    assert label is not None
    assert name is not None

    datum = datum_pb2.Datum()
    datum.width, datum.height, datum.channels = img.shape
    datum.data = img.tostring()
    datum.label = label
    datum.name = bytes(name, encoding='utf8')
    return datum


def array_to_datum_test(name, img):
    assert img.ndim == 3
    assert img.dtype == np.uint8
    datum = datum_pb2.Datum()
    datum.width, datum.height, datum.channels = img.shape
    datum.data = img.tostring()
    datum.name = bytes(name, encoding='utf8')
    return datum


def save_train_lmdb(imgs_dir, labels_dir):
    new_w, new_h = 48, 32
    save_train_db = "/data-private/face/wx_space/label_mark/gaze_label_mark/train_db/"
    save_valid_db = "/data-private/face/wx_space/label_mark/gaze_label_mark/valid_db/"
    for date in os.listdir(labels_dir):
        if date >"11.12.txt":
            save_train_path = save_train_db + date.replace("txt", "db")
            save_valid_path = save_valid_db + date.replace("txt", "db")
            imgs_path = os.path.join(imgs_dir, date.replace(".txt", ""))
            lines = []
            label_path = os.path.join(labels_dir, date)
            with open(label_path, "r") as lp:
                lines += lp.readlines()

            Num = len(lines)
            Num_train = int(Num * 0.8)
            np.random.seed(1)
            index = np.random.permutation(Num)

            db = lmdb.open(save_train_path, map_size=1024 ** 4)
            txn = db.begin(write=True)
            # if int(txn.get("num_samples".encode())):
            #     num_samples = int(txn.get("num_samples".encode()))
            # else:
            #     num_samples = 0
            # txn.delete(key="num_samples")
            count = 0
            for i in index[:Num_train]:
                line = lines[i]
                name = line.strip().split(" ")[0]
                label = int(line.strip().split(" ")[1])
                img_name = os.path.join(imgs_path, name)
                eye = cv2.imread(img_name)
                img = cv2.resize(eye, (new_w, new_h))
                datum = array_to_datum_train(name, img, label)
                txn.put("{:0>8d}".format(count).encode(), datum.SerializeToString())
                count += 1
            print('num_samples: ', count)
            txn.put('num_samples'.encode(), str(count).encode())
            txn.commit()
            db.close()

            db = lmdb.open(save_valid_path, map_size=1024 ** 4)
            txn = db.begin(write=True)
            count = 0
            for i in index[Num_train:]:
                line = lines[i]
                name = line.strip().split(" ")[0]
                label = int(line.strip().split(" ")[1])
                img_name = os.path.join(imgs_path, name)
                eye = cv2.imread(img_name)
                img = cv2.resize(eye, (new_w, new_h))
                datum = array_to_datum_train(name, img, label)
                txn.put("{:0>8d}".format(count).encode(), datum.SerializeToString())
                count += 1
            print('num_samples: ', count)
            txn.put('num_samples'.encode(), str(count).encode())
            txn.commit()
            db.close()


def merge_db(db_dir, phase):
    save_db_path = os.path.join("/data-private/face/wx_space/label_mark/gaze_label_mark/", phase)
    save_db = lmdb.open(save_db_path, map_size=1024 ** 4)
    save_txn = save_db.begin(write=True)
    Num_samples = 0
    datum = datum_pb2.Datum()
    count = 0
    for file in os.listdir(db_dir):
        db_path = os.path.join(db_dir, file)
        db = lmdb.open(db_path, map_size=1024 ** 4)
        db_txn = db.begin()
        num = int(db_txn.get("num_samples".encode()))
    #     database = db_txn.cursor()
    #     for (key, value) in database:
    #         save_txn.put(key, value)
    #         Num_samples += 1
    #     db.close()
    # print(Num_samples)
    # save_txn.put('num_samples'.encode(), str(Num_samples).encode())
    # save_txn.commit()
    # save_db.close()
        for i in range(num):
            value = db_txn.get("{:0>8d}".format(i).encode())
            datum.ParseFromString(value)
            img = np.fromstring(datum.data, dtype=np.uint8).reshape(
                datum.width, datum.height, datum.channels)
            name = datum.name.decode()
            label = datum.label
            datum_save = array_to_datum_train(name, img, label)
            save_txn.put("{:0>8d}".format(count).encode(), datum_save.SerializeToString())
            count += 1
    print('num_samples: ', count)
    save_txn.put('num_samples'.encode(), str(count).encode())
    save_txn.commit()
    save_db.close()

# def save_train_lmdb(imgs_path, labels_path, save_train_path, save_valid_path):
#     new_w, new_h = 48, 32
#     lines = []
#     with open(labels_path, "r") as lp:
#         lines += lp.readlines()
#
#     Num = len(lines)
#     Num_train = int(Num * 0.8)
#     np.random.seed(1)
#     index = np.random.permutation(Num)
#
#     db = lmdb.open(save_train_path, map_size=1024 ** 4)
#     txn = db.begin(write=True)
#     count = 0
#     for i in index[:Num_train]:
#         line = lines[i]
#         name = line.strip().split(" ")[0]
#         label = int(line.strip().split(" ")[1])
#         img_name = os.path.join(imgs_path, name)
#         eye = cv2.imread(img_name)
#         img = cv2.resize(eye, (new_w, new_h))
#         datum = array_to_datum_train(name, img, label)
#         txn.put("{:0>8d}".format(count).encode(), datum.SerializeToString())
#         count += 1
#     print('num_samples: ', count)
#     txn.put('num_samples'.encode(), str(count).encode())
#     txn.commit()
#     db.close()
#
#     db = lmdb.open(save_valid_path, map_size=1024 ** 4)
#     txn = db.begin(write=True)
#     count = 0
#     for i in index[Num_train:]:
#         line = lines[i]
#         name = line.strip().split(" ")[0]
#         label = int(line.strip().split(" ")[1])
#         img_name = os.path.join(imgs_path, name)
#         eye = cv2.imread(img_name)
#         img = cv2.resize(eye, (new_w, new_h))
#         datum = array_to_datum_train(name, img, label)
#         txn.put("{:0>8d}".format(count).encode(), datum.SerializeToString())
#         count += 1
#     print('num_samples: ', count)
#     txn.put('num_samples'.encode(), str(count).encode())
#     txn.commit()
#     db.close()


def save_test_lmdb(imgs_path, save_test_path):
    new_w, new_h = 48, 32
    db = lmdb.open(save_test_path, map_size=1024 ** 4)
    txn = db.begin(write=True)
    count = 0
    for img_name in os.listdir(imgs_path):
        img_path = os.path.join(imgs_path, img_name)
        eye = cv2.imread(img_path)
        img = cv2.resize(eye, (new_w, new_h))
        datum = array_to_datum_test(img_name, img)
        txn.put("{:0>8d}".format(count).encode(), datum.SerializeToString())
        count += 1
    print('num_samples: ', count)
    txn.put('num_samples'.encode(), str(count).encode())
    txn.commit()
    db.close()


if __name__ == "__main__":
    imgs_path = "/data-private/face/wx_space/label_mark/img/"
    labels_path = "/data-private/face/wx_space/label_mark/gaze_label_mark/label/"
    save_test_file_path = "/data-private/face/wx_space/label_mark/gaze_label_mark/test/"
    db_dir = "/data-private/face/wx_space/label_mark/gaze_label_mark/valid_db/"
    # save_train_lmdb(imgs_path, labels_path)
    # merge_db(db_dir, "valid_1.db")
    for date in os.listdir(imgs_path):
        save_test_path = save_test_file_path + date + ".db"
        img_test_path = os.path.join(imgs_path, date)
        save_test_lmdb(img_test_path, save_test_path)

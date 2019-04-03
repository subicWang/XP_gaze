import time

import argparse
import torch
import torch.utils.data
import torchvision
import numpy as np
from lib.network import *
from lib.gaze_dataset import *
from lib.estimate_classification import EstimateClassification
import matplotlib.pyplot as plt
from XTorch.utils import cfg_from_file
from XTorch.solver import Solver


class GazeSolver(Solver):
    def __init__(self, cfg, session=None, save=True):
        super().__init__(cfg, save=save, session=session)
        self.criterion_f1, self.criterion_f2 = self._create_criterion()
        self.estimate = EstimateClassification()
        self.lr = 0.
        self.opts = {
            'iters': {'title': 'Batch Loss Trace', 'xlabel': 'Batch Number', 'ylabel': 'Loss'},
            'train': {'title': 'Train Loss', 'xlabel': 'Epoch', 'ylabel': 'Loss'},
            'val': {'title': 'Val Loss', 'xlabel': 'Epoch', 'ylabel': 'Loss'},
            'epoch': {'title': "Epoch Loss", 'xlabel': 'Epoch', 'ylabel': 'Loss', 'legend': ['train loss', 'val loss']},
            'Train_error': {'title': "Train error", 'xlabel': 'Epoch', 'ylabel': 'Error', 'legend': ['yaw', 'pitch']},
            'Val_error': {'title': "Val error", 'xlabel': 'Epoch', 'ylabel': 'Error', 'legend': ['yaw', 'pitch']}
        }

    def build_model(self):
        model = eval(self.cfg.MODEL)()(self.cfg)
        self.use_gpu = self.cfg.USE_GPU and torch.cuda.is_available()
        if 'RESUME' in self.cfg:
            print('resume from {}'.format(self.cfg.RESUME))
            check_point = torch.load(self.cfg.RESUME)
            model.load_state_dict(check_point['model'])
        if self.use_gpu:
            model = model.cuda()
            if self.cfg.MULTI_GPU:
                model = torch.nn.DataParallel(model)
        return model

    def _create_loader(self):
        # train_data = eval(self.cfg.DATASET)(
        #     self.cfg.TRAIN.DATA_DIR, self.cfg.TRAIN.DATA_PATH, self.cfg.IF_HEAD, self.cfg.IF_FACE)
        # val_data = eval(self.cfg.DATASET)(
        #     self.cfg.VAL.DATA_DIR, self.cfg.VAL.DATA_PATH, self.cfg.IF_HEAD, self.cfg.IF_FACE)
        train_data = eval(self.cfg.DATASET)(self.cfg.TRAIN.DATA_DIR, 48, 32, phase="train")
        val_data = eval(self.cfg.DATASET)(self.cfg.VAL.DATA_DIR, 48, 32, phase="valid")
        test_data = eval(self.cfg.DATASET)(self.cfg.TEST.DATA_DIR, 48, 32, phase="test")

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.cfg.TRAIN.BATCH_SIZE, shuffle=True,
        num_workers=self.cfg.TRAIN.NUM_WORKERS, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.cfg.VAL.BATCH_SIZE, shuffle=False,
        num_workers=self.cfg.VAL.NUM_WORKERS, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.cfg.TEST.BATCH_SIZE, shuffle=False,
        num_workers=self.cfg.TEST.NUM_WORKERS, pin_memory=True)
        return train_loader, val_loader, test_loader

    def _run_epoch_OneEye(self, epoch, model, optim, dataloader, phase='Train', iters=None):
        if phase == 'Train':
            model.train()
        else:
            model.eval()
        total_loss = 0.
        Num_tp = [0, 0, 0]
        Num_gt = [0, 0, 0]
        Num_p = [0, 0, 0]

        total_angle_error0 = np.zeros(2)
        total_angle_error1 = np.zeros(2)
        start = time.time()
        Num_select = 0
        for batch_index, data in enumerate(dataloader):
            if iters and batch_index > iters:
                break
            if self.use_gpu:
                data = [d.cuda() if not isinstance(d, list) else d for d in data]
            if phase == 'Train':
                optim.zero_grad()

            if self.cfg.IF_FACE:
                assert self.cfg.IF_HEAD
                img, gaze, headpose, face = data
                p_gaze = model(img, headpose, face)
            elif self.cfg.IF_HEAD:
                img, gaze, headpose = data
                p_gaze = model(img, headpose)
            else:
                # img, gaze = data
                img, gaze, label, name = data

            batchNum = len(img)

            p_gaze, p_label = model(img)
            # self.logger.show_images(img)
            t_label = label
            if phase == "Train":
                # p_gaze_select = p_gaze
                # gaze_select = gaze[:, :2]
                index_label = torch.eq(t_label, 1)
                # index_label = torch.eq(t_label, 2) + torch.eq(t_label, 1)
                p_gaze_select = p_gaze[index_label]
                gaze_select = gaze[:, :2][index_label]
            else:
                index_label = torch.eq(t_label, 1)
                p_gaze_select = p_gaze[index_label]
                gaze_select = gaze[:, :2][index_label]

            num_s = len(p_gaze_select)
            Num_select += num_s

            loss_label = self.criterion_f1(p_label, t_label)
            loss_reg = self.criterion_f2(p_gaze_select, gaze_select)
            if epoch > 40:
                loss = loss_label + loss_reg
            else:
                loss = loss_reg

            total_loss += loss.data.item()*num_s
            total_angle_error0 += torch.abs(p_gaze_select - gaze_select).sum(dim=0).detach().cpu().numpy()
            total_angle_error1 += torch.abs(p_gaze_select - gaze_select).mean(dim=0).detach().cpu().numpy()

            # label预测正确的个数
            t_label_cpu = t_label.detach().cpu().numpy()
            p_label_cpu = p_label.detach().cpu().numpy()
            # for i in range(batchNum):
            #     t = t_label_cpu[i]
            #     p = int(np.argmax(p_label_cpu[i]))
            #     if t == p:
            #         Num_tp[t] += 1
            #     Num_gt[t] += 1
            #     Num_p[p] += 1

            if phase == 'Train':
                loss.backward()
                optim.step()
        end = time.time()
        avg_loss = total_loss / Num_select
        avg_angle_err0 = total_angle_error0 / Num_select
        avg_angle_err1 = total_angle_error1 / len(dataloader)
        # Precision = np.array(Num_tp)/np.array(Num_p)
        # Recall = np.array(Num_tp)/np.array(Num_gt)
        self.logger.log("[{4}] epoch: {0}, loss: {1}, LR: {2}, error0: {5}, error1: {6}, time: {3}".format(
            epoch, avg_loss, self.lr, (end - start), phase, avg_angle_err0, avg_angle_err1))#
    def _run_epoch_TwoEyes(self, epoch, model, optim, dataloader, phase='Train', iters=None):
        if phase == 'Train':
            model.train()
        else:
            model.eval()
        total_left_loss = 0.
        total_right_loss = 0.
        Num_left_tp = [0, 0, 0]
        Num_right_tp = [0, 0, 0]
        Num_left_gt = [0, 0, 0]
        Num_right_gt = [0, 0, 0]
        Num_left_p = [0, 0, 0]
        Num_right_p = [0, 0, 0]


        total_left_angle_error = np.zeros(2)
        total_right_angle_error = np.zeros(2)
        start = time.time()
        Num_left_select = 0
        Num_right_select = 0

        for batch_index, data in enumerate(dataloader):
            if iters and batch_index > iters:
                break
            if self.use_gpu:
                data = [d.cuda() if not isinstance(d, list) else d for d in data]
            if phase == 'Train':
                optim.zero_grad()

            if self.cfg.IF_FACE:
                assert self.cfg.IF_HEAD
                img, gaze, headpose, face = data
                p_gaze = model(img, headpose, face)
            elif self.cfg.IF_HEAD:
                img, gaze, headpose = data
                p_gaze = model(img, headpose)
            else:
                # img, gaze = data
                img, gaze, label, name = data

            batchNum = len(img)
            t_gaze_left = gaze[:, :2]
            t_gaze_right = gaze[:, 2:]
            t_label_left = label[:, 0]
            t_label_right = label[:, 1]
            p_gaze, p_label = model(img)
            p_gaze_left = p_gaze[:, :2]
            p_gaze_right = p_gaze[:, 2:]
            p_label_left = p_label[:, :3]
            p_label_right = p_label[:, 3:]
            # 只要有一个眼睛是好的就可以训练
            if phase == "train":
                # p_gaze_left_select = p_gaze_left
                # p_gaze_right_select = p_gaze_right
                # t_gaze_left_select = t_gaze_left
                # t_gaze_right_select = t_gaze_right

                index_label = torch.eq(t_label_left, 2) + torch.eq(t_label_left, 1) + torch.eq(t_label_right,
                                                                                               2) + torch.eq(
                    t_label_right, 1)
                index_label = torch.gt(index_label, 0)

                p_gaze_left_select = p_gaze_left[index_label]
                p_gaze_right_select = p_gaze_right[index_label]
                t_gaze_left_select = t_gaze_left[index_label]
                t_gaze_right_select = t_gaze_right[index_label]
            else:
                index_label = torch.eq(t_label_left, 2) + torch.eq(t_label_left, 1) + torch.eq(t_label_right,2) + torch.eq(t_label_right, 1)
                index_label = torch.gt(index_label, 0)

                p_gaze_left_select = p_gaze_left[index_label]
                p_gaze_right_select = p_gaze_right[index_label]
                t_gaze_left_select = t_gaze_left[index_label]
                t_gaze_right_select = t_gaze_right[index_label]

            num_left = len(p_gaze_left_select)
            num_right = len(p_gaze_right_select)
            Num_left_select += num_left
            Num_right_select += num_right

            loss_label_left = self.criterion_f1(p_label_left, t_label_left)
            loss_label_right = self.criterion_f1(p_label_right, t_label_right)
            loss_reg_left = self.criterion_f2(p_gaze_left_select, t_gaze_left_select)
            loss_reg_right = self.criterion_f2(p_gaze_right_select, t_gaze_right_select)

            loss_left = loss_label_left + loss_reg_left
            loss_right = loss_label_right + loss_reg_right
            loss = loss_left + loss_right
            if phase == 'Train':
                loss.backward()
                optim.step()

            total_left_loss += loss_left.data.item()*num_left
            total_right_loss += loss_right.data.item()*num_right
            total_left_angle_error += torch.abs(p_gaze_left_select - t_gaze_left_select).sum(dim=0).detach().cpu().numpy()
            total_right_angle_error += torch.abs(p_gaze_right_select - t_gaze_right_select).sum(dim=0).detach().cpu().numpy()

            # label预测正确的个数
            t_label_left_cpu = t_label_left.detach().cpu().numpy()
            t_label_right_cpu = t_label_right.detach().cpu().numpy()
            p_label_left_cpu = p_label_left.detach().cpu().numpy()
            p_label_right_cpu = p_label_right.detach().cpu().numpy()
            for i in range(batchNum):
                t_left = t_label_left_cpu[i]
                t_right = t_label_right_cpu[i]
                p_left = int(np.argmax(p_label_left_cpu[i]))
                p_right = int(np.argmax(p_label_right_cpu[i]))
                if t_left == p_left:
                    Num_left_tp[t_left] += 1
                if t_right == p_right:
                    Num_right_tp[t_right] += 1
                Num_left_gt[t_left] += 1
                Num_right_gt[t_right] += 1
                Num_left_p[p_left] += 1
                Num_right_p[p_right] += 1


        end = time.time()
        avg_left_loss = total_left_loss / Num_left_select
        avg_right_loss = total_right_loss / Num_right_select
        avg_left_angle_err = total_left_angle_error / Num_left_select
        avg_right_angle_err = total_right_angle_error / Num_right_select
        Precision = (np.array(Num_left_tp) + np.array(Num_right_tp))/(np.array(Num_left_p)+np.array(Num_right_p))
        Recall = (np.array(Num_left_tp) + np.array(Num_right_tp))/(np.array(Num_left_gt)+np.array(Num_right_gt))
        self.logger.log("[{9}] epoch: {0}, loss_left: {1}, loss_right: {2},  error_left: {3},  error_right: {4}, \
         P: {5}, R: {6}, time: {7}, LR: {8},".format(epoch, avg_left_loss, avg_right_loss, avg_left_angle_err, \
        avg_right_angle_err, Precision, Recall, (end - start), self.lr,  phase))#


    def train(self):
        epoch_size = self.cfg.TRAIN.EPOCH
        for epoch in range(self.start_epoch, epoch_size):
            if self.lr_scheduler:
                self.lr_scheduler.step(epoch)
                self.lr = self.lr_scheduler.get_lr()[0]

            self._train_epoch(epoch, self.model, self.optim, self.train_loader, self.lr_scheduler)
            self._val_epoch(epoch, self.model, self.optim, self.val_loader, self.lr_scheduler)

            if (epoch + 1) % 10 == 0:
                self.logger.save_model(self.model, self.optim, self.lr_scheduler, epoch)
    def test(self):
        self._test_epoch(self.model, self.test_loader)
    def _train_epoch(self, epoch, model, optim, dataloader, lr_scheduler, iters=None):
        self._run_epoch_OneEye(epoch, model, optim, dataloader, phase='Train')
        # self._run_epoch_TwoEyes(epoch, model, optim, dataloader, phase='Train')

    def _val_epoch(self, epoch, model, optim, dataloader, lr_scheduler):
        self._run_epoch_OneEye(epoch, model, optim, dataloader, phase='Val')
        # self._run_epoch_TwoEyes(epoch, model, optim, dataloader, phase='Val')

    def _test_epoch(self, model, test_loader):
        model.eval()
        start = time.time()
        Estimate = np.zeros((51, 2))
        Estimate_index = np.zeros((51, 2))
        for batch_index, data in enumerate(test_loader):
            if self.use_gpu:
                data = [d.cuda() if not isinstance(d, list) else d for d in data]
            img, gaze = data
            # img, gaze, label, name = data
            p_gaze, plabel = model(img)
            # self.logger.show_images(img)
            p_index = []
            for i in range(len(plabel)):
                if torch.eq(torch.argmax(plabel[i]), 0):
                    pass
                else:
                    p_index.append(i)


            # p_gaze_select = p_gaze[p_index]
            # gaze_select = gaze[p_index, :2]
            # angle_error = torch.abs(p_gaze_select - gaze_select[:, :2]).detach().cpu().numpy()
            # e, e_index = self.estimate(gaze[p_index].detach().cpu().numpy(), angle_error)

            angle_error = torch.abs(p_gaze - gaze[:, :2]).detach().cpu().numpy()
            e, e_index = self.estimate(gaze.detach().cpu().numpy(), angle_error)
            Estimate += e
            Estimate_index += e_index
        Avg_Est = Estimate/(Estimate_index+0.001)
        # print(Avg_Est)
        end = time.time()
        self.logger.plt_save_grah(Avg_Est[:3], title="predict_landmark glass", ylabel="Mean Error", labels=["NO", "Yanjing", "Mojing"],
                                  save_name="predict_landmark glass.jpg")
        env_name = "estimate_dt"
        self.logger.show_graph(Avg_Est[:3], "glass", env_name)
        object = ["gaze_x", "gaze_y", "pitch", "yaw"]
        for i in range(4):
            self.logger.show_graph(Avg_Est[3+i*12:3+(i+1)*12], object[i], env_name)
            self.logger.plt_save_grah(Avg_Est[3+i*12:3+(i+1)*12], title="predict_landmark "+object[i], ylabel="Mean Error",
                                      labels=["-60", "-50", "-40", "-30", "-20", "-10", "0", "10", "20", "30", "40", "50"],
                                      save_name="predict_landmark "+ object[i]+".jpg")

    def _create_criterion(self):
        f1 = torch.nn.CrossEntropyLoss()
        f2 = torch.nn.MSELoss()
        if self.use_gpu:
            f1 = f1.cuda()
            f2 = f2.cuda()
        return f1, f2


if "__main__" == __name__:
    cfg = cfg_from_file("cfgs/gaze_small4_one-eyed.yml")
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(["6,7"])
    solver = GazeSolver(cfg)
    solver.train()

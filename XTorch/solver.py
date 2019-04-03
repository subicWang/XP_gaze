import torch

import yaml
from abc import abstractmethod

from XTorch.logger import Logger
from XTorch.scheduler import SchedulerBuilder, FindLR
from XTorch.utils import AttrDict


class Solver(object):
    def __init__(self, cfg, session=None, save=True):
        # with open(cfg, 'r') as f:
        #     self.cfg = AttrDict(yaml.load(f.read()))
        self.cfg = cfg
        self.use_gpu = False
        self.model = self.build_model()
        self.optim = self._create_optim(self.model.parameters())
        self.lr_scheduler = self._create_lr_scheduler()
        self.train_loader, self.val_loader, self.test_loader = self._create_loader()
        self.start_epoch = 0

        # if hasattr(self.cfg, 'RESUME'):
        if 'RESUME' in self.cfg:
            # print('resume from {}'.format(self.cfg.RESUME))
            check_point = torch.load(self.cfg.RESUME)
            if not session:
                session = check_point['session']
            # self.optim.load_state_dict(check_point['optimizer'])
            # self.lr_scheduler.load_state_dict(check_point['scheduler'])
            self.start_epoch = check_point['epoch']
        if 'SESSION' in self.cfg:
            session = self.cfg.SESSION
        self.logger = Logger(self.cfg.NAME, self.cfg.SAVE_PATH, self.cfg.VISDOM, self.cfg.PORT, save, session)
        self.logger.log_config(self.cfg)
        self.find_lr_opts = {
            'loss_lr': {'title': 'Find learning rate', 'xlabel': 'lr', 'ylabel': 'loss', 'xtype': 'log'},
        }

    def _create_lr_scheduler(self):
        builder = SchedulerBuilder()
        return builder(self.optim, self.cfg.TRAIN.OPTIMIZATION.SCHEDULER)

    def _create_optim(self, params, lr=None):
        optim_config = self.cfg.TRAIN.OPTIMIZATION
        if lr is None:
            lr = self.cfg.TRAIN.LR
        if optim_config.TYPE == 'SGD':
            return torch.optim.SGD(
                params, lr, optim_config.MOMENTUM,
                weight_decay=optim_config.WEIGHT_DECAY)
        elif optim_config.TYPE == 'ADAM':
            return torch.optim.Adam(params=params,
                                    lr=lr,
                                    weight_decay=optim_config.WEIGHT_DECAY)
        else:
            return None

    def find_lr(self, iters=100):
        batch_num = iters // 20
        # epochs = iters // len(self.train_loader) + 1
        model = self.build_model()
        optim = self._create_optim(model.parameters(), 1e-6)
        # lr_scheduler = self._create_lr_scheduler()
        scheduler = FindLR(optim)
        # scheduler = torch.optim.lr_scheduler.StepLR(optim, 10, 0.1)

        data_loader, _, _ = self._create_loader()
        for epoch in range(20):
            # scheduler.step()
            loss = self._train_epoch(epoch, model, optim, data_loader, scheduler, batch_num)
            lr = scheduler.get_lr()[0]
            self.logger.line('loss', loss, lr, win='flr', opts=self.find_lr_opts['loss_lr'])
            # self.logger.line('lr', lr, epoch, win='flr', opts=self.find_lr_opts['lr'])

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train(self):
        pass

    # @abstractmethod
    # def test(self):
    #     pass

    @abstractmethod
    def _train_epoch(self, epoch, model, optim, dataloader, lr_scheduler, iters=None):
        pass

    @abstractmethod
    def _create_loader(self):
        pass

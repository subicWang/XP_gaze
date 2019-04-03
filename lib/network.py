from abc import abstractmethod


import models.smallNet
import models.eye_model


class GazeNetInterface(object):
    @abstractmethod
    def __call__(self, cfg):
        pass


class ResNet18(GazeNetInterface):
    def __call__(self, cfg):
        return models.resnet18.ResNet18()


class ResNet18Ori(GazeNetInterface):
    def __call__(self, cfg):
        return models.resnet18_ori.Resnet18Ori()


class ResNet10Ori(GazeNetInterface):
    def __call__(self, cfg):
        return models.resnet10_ori.Resnet10Ori()


class ResNet10_Small(GazeNetInterface):
    def __call__(self, cfg):
        return models.resnet10_smaller.ResNet()


class EyeDistinguish(GazeNetInterface):
    def __call__(self, cfg):
        return models.eye_model.EyeDistinguish()


class SmallNet(GazeNetInterface):
    def __call__(self, cfg):
        return models.smallNet.SmallNet1()


class SmallNet2(GazeNetInterface):
    def __call__(self, cfg):
        return models.smallNet.SmallNet2()


class SmallNet3(GazeNetInterface):
    def __call__(self, cfg):
        return models.smallNet.SmallNet3()


class SmallNet4(GazeNetInterface):
    def __call__(self, cfg):
        return models.smallNet.SmallNet4()


class SmallNet4OneEye(GazeNetInterface):
    def __call__(self, cfg):
        return models.smallNet.SmallNet4OneEye()


class SmallNet4TwoEyes(GazeNetInterface):
    def __call__(self, cfg):
        return models.smallNet.SmallNet4TwoEyes()


class SqueezeNet(GazeNetInterface):
    def __call__(self, cfg):
        return models.squeezeNet.squeezenet1_0()


class ResNet18Trimmed(GazeNetInterface):
    def __call__(self, cfg):
        return models.resnet18_trimmed.ResNet18Trimmed()


class ResNet18NoHp(GazeNetInterface):
    def __call__(self, cfg):
        return models.resnet18_nohp.ResNet18()

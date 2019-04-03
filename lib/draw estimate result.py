# -*- coding: utf-8 -*-
"""
Aouther: Subic
Time: 2019/3/5: 16:37
"""
import numpy as np
from XTorch.logger import Logger
import matplotlib.pyplot as plt
Estimate = np.array([[ 4.71299932,  4.14205514],[ 7.63983126,  6.04581187],[13.82303224, 10.63044182],[ 8.1595518,   6.15837799],
[ 7.04065606,  5.23999012], [ 6.06505386,  4.26975969],[ 7.19621127, 6.08562753], [ 8.10778114,  5.92843897],[ 8.8591747,   7.27509959],
[ 6.96372587,  6.43850999],[ 7.96640837,  6.45671985], [ 6.64811206,  5.94655891],[ 6.16041961,  5.10406837],[ 9.99629694,  7.86037778],
[7.89168773, 6.37529935], [8.2261595,   6.9716534]])
# self.logger.plt_save_grah(Avg_Est[:3], title="is wear glass", ylabel="Mean Error", labels=["NO", "Yanjing", "Mojing"],
#                           save_name="is wear glass.jpg")
log = Logger("GazeDetection", "../check_points/", True, 10011)
# bins = list(Estimate[:3].reshape(-1, 1))

# log.plt_save_grah(x, title="is wear glass", ylabel="Mean Error", labels=["NO", "Yanjing", "Mojing"],
#                                   save_name="is wear glass.jpg")


# labels = ["NO", "Yanjing", "Mojing"]
labels = ["-60", "-50", "-40", "-30", "-20", "-10", "0", "10", "20", "30", "40", "50"]
save_name ="is wear glass.jpg"

X1 = Estimate[3:, 0]
X2 = Estimate[3:, 1]
N = len(X1)
a = plt.bar(np.arange(N), X1, width=0.3)
b = plt.bar(np.arange(N)+0.3, X2, width=0.3)
plt.xticks(np.arange(N)+0.15, labels)
for rect in a:
    height = rect.get_height()
    plt.text(rect.get_x()+0.02, 1.03*height, '%.2f' % float(height))
for rect in b:
    height = rect.get_height()
    plt.text(rect.get_x()+0.02, 1.03*height, '%.2f' % float(height))
# plt.xlabel(xlabel)
# plt.ylabel(ylabel)
# plt.title(title)
plt.savefig(save_name)
# print(np.arange(1, 6, 2))



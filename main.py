from network import GetDiffNet
from dataset import dateMaker
from chainer import optimizers
import numpy as np

DM = dateMaker()

Net = GetDiffNet()
Net.zerograds()
opt = optimizers.Adam()
opt.setup(Net)

y = np.ones((10, 1), dtype=np.float32)
for data in DM:
    loss = Net(data, y)
    print(loss)
    loss.backward()
    opt.update()
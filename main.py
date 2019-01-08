from network import GetDiffNet
from dataset import dateMaker

DM = dateMaker()

Net = GetDiffNet()
for data in DM:
    loss = Net(data, )
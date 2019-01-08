from chainer import Chain
from chainer import links as L
from chainer import functions as F
import numpy as np

class GetDiffNet(Chain):
    def __init__(self):
        super(GetDiffNet, self).__init__()
        with self.init_scope():
            self.nn1 = L.Linear(None, 200)
            self.nn2 = L.Linear(200, 200)
            self.nn3 = L.Linear(200, 200)
            self.decoder = (200, 1)
            self.cnn1 = L.Convolution2D(None, 3)
            self.cnn2 = L.Convolution2D(3, 6)
            self.cnn3 = L.Convolution2D(6, 12)
    
    def __call__(self, x, y):
        layer = []
        out = F.relu(self.nn1(x))
        out = out.ljust(200, '0')
        layer.append(out)
        out = F.relu(self.nn2(out))
        layer.append(out)
        out = self.nn3(out)
        layer.append(out)
        out = self.decoder(out)
        return F.mean_squared_error(out, y)
        


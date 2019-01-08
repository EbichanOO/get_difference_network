from chainer import Chain, Variable
from chainer import links as L
from chainer import functions as F
import numpy as np

class GetDiffNet(Chain):
    def __init__(self):
        super(GetDiffNet, self).__init__()
        with self.init_scope():
            self.nn1 = L.Linear(None, 400)
            self.nn2 = L.Linear(400, 400)
            self.nn3 = L.Linear(400, 400)
            self.decoder = L.Linear(400, 1)
            self.cnn1 = L.Convolution2D(3, 6)
            self.cnn2 = L.Convolution2D(6, 12)
            self.cnn3 = L.Convolution2D(12, 12)
    
    def __call__(self, x, y):
        #layer = Variable()
        out = F.relu(self.nn1(x))
        layer = out
        out = F.relu(self.nn2(out))
        layer = F.concat((layer, out))
        out = self.nn3(out)
        layer = F.concat((layer, out))
        out = self.decoder(out)
        cate = F.reshape(layer, (10, 3, 20, 20))
        cate = self.cnn1(cate)
        cate = self.cnn2(cate)
        #cate = self.cnn3(cate)
        return F.mean_squared_error(out, y)
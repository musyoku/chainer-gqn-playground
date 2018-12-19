import chainer
import chainer.functions as cf
import chainer.links as nn
from chainer.initializers import HeNormal


class SubPixelConvolutionUpsampler(chainer.Chain):
    def __init__(self, channels, scale):
        super().__init__()
        self.scale = scale
        with self.init_scope():
            self.conv = nn.Convolution2D(
                None,
                channels,
                ksize=3,
                stride=1,
                pad=1,
                initialW=HeNormal(0.1))

    def __call__(self, x):
        return cf.depth2space(self.conv(x), r=self.scale)


class DeconvolutionUpsampler(chainer.Chain):
    def __init__(self, channels):
        super().__init__()
        with self.init_scope():
            self.deconv = nn.Deconvolution2D(
                None,
                channels,
                ksize=4,
                stride=4,
                pad=0,
                initialW=HeNormal(0.1))

    def __call__(self, x):
        return self.deconv(x)

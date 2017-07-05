import numpy as np
from dataset import *
import sys, os


trans = ObservationTransformer()
x = np.load('obs.npy')
x = trans.transform(x)
x = np.reshape(x, [81, 6])
print x.shape


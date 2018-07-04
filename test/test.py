# -*- coding: utf-8 -*-
# @Time    : 7/4/2018 11:54 AM
# @Author  : sunyonghai
# @File    : test.py
# @Software: ZJ_AI
import numpy as np

def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]
c = np.ones((100,))
a = make_batches(100, 12)
b = slice(a[0][0], a[0][1])
print(c[b])
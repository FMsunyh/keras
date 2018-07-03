# -*- coding: utf-8 -*-
# @Time    : 6/29/2018 4:22 PM
# @Author  : sunyonghai
# @File    : load_data_test.py
# @Software: ZJ_AI
import datasets.cifar10

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data(test_split=0.1)
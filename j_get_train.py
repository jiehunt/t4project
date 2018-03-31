import os
import time
import warnings
import numpy as np
import pandas as pd
import sys, re, csv, codecs
import string
import requests, re, sys
import logging
import psutil
import glob
import random
import gc

from scipy.sparse import hstack
from scipy.sparse import csr_matrix
from scipy.stats  import rankdata


def h_get_train():
    DATA_NUMBER = 40000000
    path_train ='./input/train.csv'
    path_test = './input/test.csv'
    dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint16',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32'
            }
    df = pd.read_csv(path_train, dtype=dtypes)
    df.drop(df[df['is_attributed'] <1 ].index, inplace=True)
    print (df.describe())
    path_train_1 ='./input/train_1.csv'
    df.to_csv(path_train_1, index=False)
    del df
    gc.collect()

    df = pd.read_csv(path_train, dtype=dtypes)
    df.drop(df[df.is_attributed > 0 ].index, inplace=True)
    print (df.describe())
    path_train_0 ='./input/train_all0.csv'
    df.to_csv(path_train_0, index=False)
    del df
    gc.collect()

    df = pd.read_csv(path_train_0, dtype=dtypes)
    number = len(df)
    row_list = random.sample(range(number-1), DATA_NUMBER)
    df_2 = df.iloc[row_list]
    path_train_2 ='./input/train_0.csv'
    df_2.to_csv(path_train_2, index=False)

    return

def h_get_zero():
    DATA_NUMBER = 60000000
    path_train ='./input/train.csv'
    path_test = './input/test.csv'
    dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint16',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32'
            }

    path_train_0 ='./input/train_all0.csv'
    df = pd.read_csv(path_train_0, dtype=dtypes)

    list_row = range(0,len(df),2)
    data0 = df.iloc[list_row]
    path_train_1 ='./input/train_001.csv'
    data0.to_csv(path_train_1, index=False)
    print (len(data0))
    del data0
    gc.collect()

    list_row = range(1,len(df),2)
    data0 = df.iloc[list_row]
    path_train_1 ='./input/train_002.csv'
    data0.to_csv(path_train_1, index=False)
    print (len(data1))
    del data0
    gc.collect()

    # list_row = range(0,DATA_NUMBER)
    # data0 = df.iloc[list_row]
    # path_train_1 ='./input/train_00.csv'
    # data0.to_csv(path_train_1, index=False)
    # print (len(df))
    # del data0
    # gc.collect()

    # list_row = range(DATA_NUMBER,2*DATA_NUMBER)
    # data0 = df.iloc[list_row]
    # path_train_1 ='./input/train_01.csv'
    # data0.to_csv(path_train_1, index=False)
    # del data0
    # gc.collect()

    # list_row = range(DATA_NUMBER*2,len(df))
    # data0 = df.iloc[list_row]
    # path_train_1 ='./input/train_02.csv'
    # data0.to_csv(path_train_1, index=False)
    # del data0
    # gc.collect()
    return


if __name__ == '__main__':
    # h_get_train()
    h_get_zero()

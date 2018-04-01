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

from contextlib import contextmanager

@contextmanager
def timer(name):
    """
    Taken from Konstantin Lopuhin https://www.kaggle.com/lopuhin
    in script named : Mercari Golf: 0.3875 CV in 75 LOC, 1900 s
    https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
    """
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


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
    print (len(data0))
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

def h_get_pseudo_data(path_sub):
    path_test = './input/test.csv'
    test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
    sub_cols = ['is_attributed']

    dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint16',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32'
            }

    with timer('Loading the submission data...'):
        sub = pd.read_csv(path_sub, header=0, usecols=sub_cols)

    sub['is_attributed'] = sub['is_attributed'].apply(lambda x: 1 if x >= 0.5 else 0 )

    with timer('Loading the test data...'):
        test = pd.read_csv(path_test, dtype=dtypes, header=0, usecols=test_cols)

    pseudo_df = pd.concat([test, sub],axis=1)
    print (pseudo_df.info())
    print (pseudo_df.head())

    outfile = 'pseudo/' + 'pseudo.csv'
    pseudo_df.to_csv(outfile, index=False, float_format="%.6f")

    return

if __name__ == '__main__':
    # h_get_train()
    # h_get_zero()
    with timer('goto prepare pseudo file ... '):
        path_sub = 'output/set20lgbandy_org.csv'
        h_get_pseudo_data(path_sub)


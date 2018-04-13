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

from scipy.sparse import hstack
from scipy.sparse import csr_matrix
from scipy.stats  import rankdata

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectFromModel

from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K

from keras.layers import Dense, Input, Embedding, Dropout, Activation, Conv1D
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Bidirectional
from keras.layers import BatchNormalization
from keras.layers import GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D

from keras.engine import InputSpec, Layer
from keras.models import Model
from keras.models import load_model

from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

from keras.optimizers import Adam, RMSprop

from bayes_opt import BayesianOptimization
from contextlib import contextmanager

from collections import defaultdict

import gc
import lightgbm as lgb
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import csv

""""""""""""""""""""""""""""""
# system setting
""""""""""""""""""""""""""""""
warnings.filterwarnings('ignore')
os.environ["OMP_NUM_THREADS"] = "4"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
least_freq_hours_in_test_data = [6, 11, 15]

# t0 = datetime.datetime.now()
log_name = "aaa" + '.log'
log_file = open(log_name, 'a')

""""""""""""""""""""""""""""""
# Help Function
""""""""""""""""""""""""""""""
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

start_time = time.time()

def h_rank(predict_list):

    predictions = np.zeros_like(predict_list[0])
    for predict in predict_list:
        for i in range(6):
            predictions[:,i] = np.add( predictions[:,i], rankdata(predict.iloc[:,i])/predictions.shape[0] )

    predictions /= len(predict_list)
    return predictions

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))

def h_get_keras_data(dataset):
    # if feature_type == 'andy_org':
    #     feature_names = ['ip', 'device', 'app', 'os', 'channel', 'hour', 'n_channels', 'ip_app_count', 'ip_app_os_count']
    # elif feature_type == 'andy_doufu':
    #     feature_names = ['ip', 'device', 'app', 'os', 'channel', 'hour', 'n_channels', 'ip_app_count', 'ip_app_os_count', 'app_channel_count']

    print (type(dataset))
    columns = dataset.columns

    X = {}
    if feature_type == 'nano':
        emb_feature =  ['app','device','os', 'channel', 'hour',
              'nip_day_test_hh',  'nip_hh_os', 'nip_hh_dev']

        for name in emb_feature:
            X[str(name)] = np.array(dataset[[str(name)]])

        other_feature = list(set(columns) - set(emb_feature) )
        # X[str('other_feature')]  = np.array(dataset[other_feature])
        X[str('ip_app_nextClick')]  = np.array(dataset[[str('ip_app_nextClick')]])
        # X[str('other_feature')]  = X[str('other_feature')].reshape((1,len(dataset),len(other_feature)))
    else:
        for name in columns:
            X[str(name)] = np.array(dataset[[str(name)]])

    return X

def h_get_keras_data2(dataset):
    X  = {}
    X[str('all_feature')]  = np.array(dataset)

    return X

def h_get_train_test_list():
   oof_files= glob.glob("oof/*")
   train_list = []
   test_list = []

   for f in oof_files:
       train_list.append(f)
       oof_path = str(f).split('/')[0]
       oof_file = str(f).split('/')[1]
       # oof_path = str(f).split('\\')[0]
       # oof_file = str(f).split('\\')[1]
       oof_test_pre = str(oof_file).split('oof')[0]
       test_file = str(oof_path) + '_test/'+str(oof_test_pre) + 'oof_test.csv'
       test_list.append(test_file)

   return train_list, test_list

def h_prepare_data_train(file_list):
    class_names = ['is_attributed']
    class_names_oof = []
    for c in class_names:
        class_names_oof.append(c+'_oof')

    df = pd.read_csv('input/train.csv')
    df = df[class_names]
    for (n, f) in enumerate(file_list):
        one_file = pd.read_csv(f)
        one_file_n = one_file[class_names_oof]
        n_class_name = []
        for c in class_names_oof:
            n_class_name.append(c+str(n))

        one_file_n.columns = n_class_name
        df = pd.concat([df, one_file_n], axis=1)

    return df

def h_prepare_data_test(file_list):
    class_names = ['is_attributed']
    class_names_oof = []
    for c in class_names:
        class_names_oof.append(c+'_oof')

    df = pd.DataFrame()
    for (n, f) in enumerate(file_list):
        one_file = pd.read_csv(f)
        one_file_n = one_file[class_names]
        n_class_name = []
        for c in class_names_oof:
            n_class_name.append(c+str(n))
        one_file_n.columns = n_class_name
        df = pd.concat([df, one_file_n], axis=1)

    return df


""""""""""""""""""""""""""""""
# Feature
""""""""""""""""""""""""""""""
# Aggregation function
log_group = 100000 # 1000 views -> 60% confidence, 100 views -> 40% confidence
def rate_calculation(x):
    """Calculate the attributed rate. Scale by confidence"""
    rate = x.sum() / float(x.count())
    conf = np.min([1, np.log(x.count()) / log_group])
    return rate * conf


def f_get_train_test_data(data_set, feature_type, have_pse):

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

    # train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time','attributed_time', 'is_attributed']
    # test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time','attributed_time']
    train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
    test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']

    SKIP_ROWS = 100000000
    skip = range(1, SKIP_ROWS)

    with timer('Loading the training data...'):
        if data_set == 'set1':
            train = pd.read_csv(path_train, skiprows=skip, dtype=dtypes, header=0, usecols=train_cols)
        elif data_set == 'set0':
            train = pd.read_csv(path_train, nrows=SKIP_ROWS, dtype=dtypes, header=0, usecols=train_cols)
        elif data_set == 'setfull':
            train = pd.read_csv(path_train, dtype=dtypes, header=0, usecols=train_cols)
        elif data_set == 'set01':
            path_train ='./input/train_1.csv'
            train_1 = pd.read_csv(path_train, dtype=dtypes, header=0, usecols=train_cols)
            path_train ='./input/train_0.csv'
            train_0 = pd.read_csv(path_train, dtype=dtypes, header=0, usecols=train_cols)
            train = pd.concat([train_1, train_0])
            del train_0, train_1
            gc.collect()
        elif data_set == 'set001':
            path_train ='./input/train_1.csv'
            train_1 = pd.read_csv(path_train, dtype=dtypes, header=0, usecols=train_cols)
            path_train ='./input/train_00.csv'
            train_0 = pd.read_csv(path_train, dtype=dtypes, header=0, usecols=train_cols)
            train = pd.concat([train_1, train_0])
            del train_0, train_1
            gc.collect()
        elif data_set == 'set20':
            path_train ='./input/train_1.csv'
            train_1 = pd.read_csv(path_train, dtype=dtypes, header=0, usecols=train_cols)
            path_train ='./input/train_001.csv'
            train_0 = pd.read_csv(path_train, dtype=dtypes, header=0, usecols=train_cols)
            train = pd.concat([train_1, train_0])
            del train_0, train_1
            gc.collect()
        elif data_set == 'set21':
            path_train ='./input/train_1.csv'
            train_1 = pd.read_csv(path_train, dtype=dtypes, header=0, usecols=train_cols)
            path_train ='./input/train_002.csv'
            train_0 = pd.read_csv(path_train, dtype=dtypes, header=0, usecols=train_cols)
            train = pd.concat([train_1, train_0])
            del train_0, train_1
            gc.collect()

    with timer('Loading the test data...'):
        test = pd.read_csv(path_test, dtype=dtypes, header=0, usecols=test_cols)
        len_test = len(test)


    with timer('Binding the training and test set together...'):
        len_train = len(train)
        # print('The initial size of the train set is', len_train)
        train=train.append(test)

    with timer('Orgnizing Target Data...'):
        target = 'is_attributed'
        train.loc[train[target].isnull(),target] = 99
        train[target] = train[target].astype('uint8')
        # print (train.info())
        del test
        gc.collect()

    with timer("Creating new time features: 'hour' and 'day' and 'minute' 'second'..."):
        train['hour'] = pd.to_datetime(train.click_time).dt.hour.astype('uint8')
        train['day'] = pd.to_datetime(train.click_time).dt.day.astype('uint8')
        train['minute'] = pd.to_datetime(train.click_time).dt.minute.astype('uint8')
        train['second'] = pd.to_datetime(train.click_time).dt.second.astype('uint8')
        # train['minute'] = train['click_time'].dt.minute.astype('uint8')
        # train['second'] = train['click_time'].dt.second.astype('uint8')
        if feature_type != 'nano':
            train.drop( 'click_time', axis=1, inplace=True )
        else :
            train['click_time'] =pd.to_datetime(train.click_time)
        gc.collect()

        # train['click_hour'] = pd.to_datetime(train.attributed_time).dt.hour.astype('uint8')
        # train['click_day'] = pd.to_datetime(train.attributed_time).dt.day.astype('uint8')
    train['in_test_hh'] = (   3
                     - 2*train['hour'].isin(  most_freq_hours_in_test_data )
                     - 1*train['hour'].isin( least_freq_hours_in_test_data ) ).astype('uint8')

    gp = train[['ip', 'day', 'in_test_hh', 'channel']].groupby(by=['ip', 'day',
             'in_test_hh'])[['channel']].count().reset_index().rename(index=str,
             columns={'channel': 'nip_day_test_hh'})
    train = train.merge(gp, on=['ip','day','in_test_hh'], how='left')
    del gp
    train.drop(['in_test_hh'], axis=1, inplace=True)
    print( "nip_day_test_hh max value = ", train.nip_day_test_hh.max() )
    train['nip_day_test_hh'] = train['nip_day_test_hh'].astype('uint32')
    gc.collect()

    if feature_type == 'nano':
        ATTRIBUTION_CATEGORIES = [
            # V1 Features #
            ###############
            ['ip'], ['app'], ['device'], ['os'], ['channel'],

            # V2 Features #
            ###############
            ['app', 'channel'],
            ['app', 'os'],
            ['app', 'device'],
        ]
        # Find frequency of is_attributed for each unique value in column
        freqs = {}
        for cols in ATTRIBUTION_CATEGORIES:

            # New feature name
            new_feature = '_'.join(cols)+'_confRate'
            print (new_feature)

            # Perform the groupby
            group_object = train.groupby(cols)

            # Group sizes
            group_sizes = group_object.size()
            print(">> Calculating confidence-weighted rate for: {}.\n   Saving to: {}. Group Max /Mean / Median / Min: {} / {} / {} / {}".format(
                cols, new_feature,
                group_sizes.max(),
                np.round(group_sizes.mean(), 2),
                np.round(group_sizes.median(), 2),
                group_sizes.min()
            ))

            # Perform the merge
            train = train.merge(
                group_object['is_attributed']. \
                    apply(rate_calculation). \
                    reset_index(). \
                    rename(
                        index=str,
                        columns={'is_attributed': new_feature}
                    )[cols + [new_feature]],
                on=cols, how='left'
            )
            train[new_feature] = train[new_feature].astype('float32')
            del group_object
            gc.collect()
        # Define all the groupby transformations
        GROUPBY_AGGREGATIONS = [

            # V1 - GroupBy Features #
            #########################
            # Variance in day, for ip-app-channel
            {'groupby': ['ip','app','channel'], 'select': 'day', 'agg': 'var'},
            # Variance in hour, for ip-app-os
            {'groupby': ['ip','app','os'], 'select': 'hour', 'agg': 'var'},
            # Variance in hour, for ip-day-channel
            {'groupby': ['ip','day','channel'], 'select': 'hour', 'agg': 'var'},
            # Count, for ip-day-hour
            {'groupby': ['ip','day','hour'], 'select': 'channel', 'agg': 'count'},
            # Count, for ip-app
            {'groupby': ['ip', 'app'], 'select': 'channel', 'agg': 'count'},
            # Count, for ip-app-os
            {'groupby': ['ip', 'app', 'os'], 'select': 'channel', 'agg': 'count'},
            # Count, for ip-app-day-hour
            {'groupby': ['ip','app','day','hour'], 'select': 'channel', 'agg': 'count'},
            # Mean hour, for ip-app-channel
            {'groupby': ['ip','app','channel'], 'select': 'hour', 'agg': 'mean'},

            # V2 - GroupBy Features #
            #########################
            # Average clicks on app by distinct users; is it an app they return to?
            {'groupby': ['app'],
             'select': 'ip',
             'agg': lambda x: float(len(x)) / len(x.unique()),
             'agg_name': 'AvgViewPerDistinct'
            },
            # How popular is the app or channel?
            {'groupby': ['app'], 'select': 'channel', 'agg': 'count'},
            {'groupby': ['channel'], 'select': 'app', 'agg': 'count'}
        ]
        for spec in GROUPBY_AGGREGATIONS:

            # Name of the aggregation we're applying
            agg_name = spec['agg_name'] if 'agg_name' in spec else spec['agg']

            # Info
            print("Grouping by {}, and aggregating {} with {}".format(
                spec['groupby'], spec['select'], agg_name
            ))

            # Unique list of features to select
            all_features = list(set(spec['groupby'] + [spec['select']]))

            # Name of new feature
            new_feature = '{}_{}_{}'.format('_'.join(spec['groupby']), agg_name, spec['select'])

            # Perform the groupby
            gp = train[all_features]. \
                groupby(spec['groupby'])[spec['select']]. \
                agg(spec['agg']). \
                reset_index(). \
                rename(index=str, columns={spec['select']: new_feature})

            # Merge back to X_train
            train = train.merge(gp, on=spec['groupby'], how='left')
            if spec['agg'] == 'count':
                train[new_feature] = train[new_feature].astype('uint32')
            else:
                train[new_feature] = train[new_feature].astype('float32')
            del gp
            gc.collect()

        print (train.info())
        GROUP_BY_NEXT_CLICKS = [
            {'groupby': ['ip']},
            {'groupby': ['ip', 'app']},
            {'groupby': ['ip', 'channel']},
            {'groupby': ['ip', 'os']},
        ]

        # Calculate the time to next click for each group
        for spec in GROUP_BY_NEXT_CLICKS:

            # Name of new feature
            new_feature = '{}_nextClick'.format('_'.join(spec['groupby']))

            # Unique list of features to select
            all_features = spec['groupby'] + ['click_time']

            # Run calculation
            print(f">> Grouping by {spec['groupby']}, and saving time to next click in: {new_feature}")
            with timer ("using ..."):
                train[new_feature] = train[all_features].groupby(spec['groupby']).click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds
                train[new_feature] = train[new_feature].astype('float32')

        train.drop( 'click_time', axis=1, inplace=True )

        HISTORY_CLICKS = {
            'identical_clicks': ['ip', 'app', 'device', 'os', 'channel'],
            'app_clicks': ['ip', 'app']
        }

        # Go through different group-by combinations
        for fname, fset in HISTORY_CLICKS.items():

            # Clicks in the past
            train['prev_'+fname] = train. \
                groupby(fset). \
                cumcount(). \
                rename('prev_'+fname)

            # Clicks in the future
            train['future_'+fname] = train.iloc[::-1]. \
                groupby(fset). \
                cumcount(). \
                rename('future_'+fname).iloc[::-1]
            train['future_'+fname] = train['future_'+fname].astype('uint32')

    if feature_type == 'pranav':
        gp = train[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day',
                 'hour'])[['channel']].count().reset_index().rename(index=str,
                 columns={'channel': 'nip_day_hh'})
        train = train.merge(gp, on=['ip','day','hour'], how='left')
        del gp
        train['nip_day_hh'] = train['nip_day_hh'].astype('uint16')
        gc.collect()

        gp = train[['ip', 'day', 'os', 'hour', 'channel']].groupby(by=['ip', 'os', 'day',
                 'hour'])[['channel']].count().reset_index().rename(index=str,
                 columns={'channel': 'nip_hh_os'})
        train = train.merge(gp, on=['ip','os','hour','day'], how='left')
        del gp
        print( "nip_hh_os max value = ", train.nip_hh_os.max() )
        train['nip_hh_os'] = train['nip_hh_os'].astype('uint16')
        gc.collect()

        gp = train[['ip', 'app', 'hour', 'day', 'channel']].groupby(by=['ip', 'app', 'day',
                 'hour'])[['channel']].count().reset_index().rename(index=str,
                 columns={'channel': 'nip_hh_app'})
        train = train.merge(gp, on=['ip','app','hour','day'], how='left')
        del gp
        print( "nip_hh_app max value = ", train.nip_hh_app.max() )
        train['nip_hh_app'] = train['nip_hh_app'].astype('uint16')
        gc.collect()

        gp = train[['ip', 'device', 'hour', 'day', 'channel']].groupby(by=['ip', 'device', 'day',
                 'hour'])[['channel']].count().reset_index().rename(index=str,
                 columns={'channel': 'nip_hh_dev'})
        train = train.merge(gp, on=['ip','device','day','hour'], how='left')
        del gp
        print( "nip_hh_dev max value = ", train.nip_hh_dev.max() )
        train['nip_hh_dev'] = train['nip_hh_dev'].astype('uint32')
        gc.collect()

    if feature_type != 'nano':
        with timer('Computing the number of channels associated with ip day hour... '):
            n_chans = train[['ip','day','hour','channel']].groupby(by=['ip','day',
                    'hour'])[['channel']].count().reset_index().rename(columns={'channel': 'n_channels'})
            train = train.merge(n_chans, on=['ip','day','hour'], how='left')
            train['n_channels'] = train['n_channels'].astype('uint16')
            del n_chans
            gc.collect()

        with timer('Computing the number of channels associated with ip app...'):
            n_chans = train[['ip','app', 'channel']].groupby(by=['ip',
                    'app'])[['channel']].count().reset_index().rename(columns={'channel': 'ip_app_count'})
            train = train.merge(n_chans, on=['ip','app'], how='left')
            train['ip_app_count'] = train['ip_app_count'].astype('uint16')
            del n_chans
            gc.collect()

        with timer('Computing the number of channels associated with ip app os...'):
            n_chans = train[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app',
                    'os'])[['channel']].count().reset_index().rename(columns={'channel': 'ip_app_os_count'})
            train = train.merge(n_chans, on=['ip','app', 'os'], how='left')
            train['ip_app_os_count'] = train['ip_app_os_count'].astype('uint16')
            del n_chans
            gc.collect()

    if feature_type == 'andy_doufu':
        with timer('Computing the IP associated with app channel...'):
            n_chans = train[['ip','app', 'channel']].groupby(by=['app',
                    'channel'])[['ip']].count().reset_index().rename(columns={'ip': 'app_channel_count'})
            train = train.merge(n_chans, on=['app', 'channel'], how='left')
            train['app_channel_count'] = train['app_channel_count'].astype('uint16')
            del n_chans
            gc.collect()

    if feature_type == 'alax':
        train[['app','device','os', 'channel', 'hour', 'day']].apply(LabelEncoder().fit_transform)

    test = train[len_train:].copy().drop( target, axis=1 )
    train = train[:len_train]

    if use_pse == True:
        path_pseudo = './pseudo/pseudo.csv'
        pseudo = pd.read_csv(path_pseudo, dtype=dtypes, header=0, usecols=['is_attributed'])
        pseudo = pseudo['is_attributed']
        new_test = test.reset_index(drop=True)
        pseudo = pd.concat ([new_test, pseudo], axis=1)
    else:
        pseudo = None

    save_file = False
    if save_file == True:
        file_path = 'input/' + str(data_set)+'_'+ str(feature_type) + '_train.csv'
        train.to_csv(file_path, index=False)
        file_path = 'input/' + str(data_set)+'_'+ str(feature_type) + '_test.csv'
        test.to_csv(file_path, index=False)

    print('The size of the test set is ', len(test))
    print('The type of the test set is ', type(test))
    print('The size of the train set is ', len(train))
    print('The tyep of the train set is ', type(train))

    return train, test, pseudo



""""""""""""""""""""""""""""""
# Model
""""""""""""""""""""""""""""""
def m_old_lgb_model(csr_trn, csr_sub, train, test, feature_type):

    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    # Set LGBM parameters
    params = {
        "objective": "binary",
        'metric': {'auc'},
        "boosting_type": "gbdt",
        "verbosity": -1,
        "num_threads": 4,
        "bagging_fraction": 0.8,
        "feature_fraction": 0.8,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "verbose": -1,
        "min_split_gain": .1,
        "reg_alpha": .1,
        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        "max_bin": 63
    }
    splits = 5
    # print (type(train)) frame.DataFrame

    # Now go through folds
    # I use K-Fold for reasons described here :
    # https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/49964
    with timer("Scoring Light GBM"):
        scores = []
        folds = StratifiedKFold(n_splits=splits, shuffle=True, random_state=1)
        lgb_round_dict = defaultdict(int)
        trn_lgbset = lgb.Dataset(csr_trn, free_raw_data=False)
        # del csr_trn
        # gc.collect()

        pred = np.zeros( shape=(len(test), len(class_names)) )
        pred =pd.DataFrame(pred)
        pred.columns = class_names

        for class_name in class_names:
            print("Class %s scores : " % class_name)
            class_pred = np.zeros(len(train))
            train_target = train[class_name]
            trn_lgbset.set_label(train_target.values)

            lgb_rounds = 500

            for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train, train_target)):
                watchlist = [
                    trn_lgbset.subset(trn_idx),
                    trn_lgbset.subset(val_idx)
                ]
                # Train lgb l1
                model = lgb.train(
                    params=params,
                    train_set=watchlist[0],
                    num_boost_round=lgb_rounds,
                    valid_sets=watchlist,
                    early_stopping_rounds=50,
                    verbose_eval=0
                )
                class_pred[val_idx] = model.predict(trn_lgbset.data[val_idx], num_iteration=model.best_iteration)
                score = roc_auc_score(train_target.values[val_idx], class_pred[val_idx])
                pred[class_name] += model.predict(csr_sub, num_iteration=model.best_iteration)

                # Compute mean rounds over folds for each class
                # So that it can be re-used for test predictions
                lgb_round_dict[class_name] += model.best_iteration
                print("\t Fold %d : %.6f in %3d rounds" % (n_fold + 1, score, model.best_iteration))

            print("full score : %.6f" % roc_auc_score(train_target, class_pred))
            scores.append(roc_auc_score(train_target, class_pred))
            train[class_name + "_oof"] = class_pred

        # Save OOF predictions - may be interesting for stacking...
        file_name = 'oof/'+ 'lgb_'+str(feature_type) + '_oof.csv'
        train[["id"] + class_names + [f + "_oof" for f in class_names]].to_csv(file_name,
                                                                               index=False,
                                                                               float_format="%.8f")

        print('Total CV score is {}'.format(np.mean(scores)))

        pred = pred/splits
        return pred

def m_lgb_model(train, test, model_type, feature_type, data_type, use_pse,pseudo):

    predictors = ['device', 'app', 'os', 'channel', 'hour', 'n_channels', 'ip_app_count', 'ip_app_os_count']
    categorical = ['app', 'device', 'os', 'channel', 'hour']
    not_use_list = ['is_attributed', 'ip']
    if feature_type == 'andy_org':
        predictors = ['device', 'app', 'os', 'channel', 'hour', 'n_channels', 'ip_app_count', 'ip_app_os_count']
    elif feature_type == 'andy_doufu':
        predictors = ['device', 'app', 'os', 'channel', 'hour', 'n_channels', 'ip_app_count', 'ip_app_os_count', 'app_channel_count']
    elif feature_type == 'pranav':
        predictors = ['app','device','os', 'channel', 'hour', 'n_channels', 'ip_app_count', 'ip_app_os_count',
              'nip_day_test_hh', 'nip_day_hh', 'nip_hh_os', 'nip_hh_app', 'nip_hh_dev']
    elif feature_type == 'nano':
        cols = train.columns
        predictors = list(set(cols) - set(not_use_list))

    target = ['is_attributed']

    params = { # andy org
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.1,
        'num_leaves': 255,
        'max_depth': 8,
        'min_child_samples': 100,
        'max_bin': 100,
        'subsample': 0.7,
        'subsample_freq': 1,
        'colsample_bytree': 0.7,
        'min_child_weight': 0,
        'subsample_for_bin': 200000,
        'min_split_gain': 0,
        'reg_alpha': 0,
        'reg_lambda': 0,
        'nthread': 4,
        'verbose': 0,
        'scale_pos_weight':403,
        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        }
    params = { ## get from andy_pranva
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric':'auc',
        'learning_rate': 0.3,
        # 'learning_rate': 0.1,
        'num_leaves': 7,  # we should let it be smaller than 2^(max_depth)
        'max_depth': 4,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'nthread': 4,
        'verbose': 0,
        'scale_pos_weight':201, # 201 or 403 because training data is extremely unbalanced
        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
    }


    one_fold = True
    print ("use one_fold ? :", str(one_fold))
    splits = 3
    if one_fold == True:
        splits = 1
        len_train = len(train)
        r = 0.1 # the fraction of the train data to be used for validation
        row_list = random.sample(range(len_train-1), round(len_train*r))
        # val = train[(len_train-round(r*len_train)):len_train]
        val = train.iloc[row_list]
        print('The size of the validation set is ', len(val))

        new_list = list(set(range(len_train-1))- set(row_list))
        train = train.iloc[new_list]
        print('The size of the train set is ', len(train))

        if use_pse == True:
            train = pd.concat([train, pseudo],axis=0)

        dtrain = lgb.Dataset(train[predictors].values, label=train['is_attributed'].values,
                              feature_name=predictors,
                              categorical_feature=categorical
                              )
        del train
        gc.collect()

        dvalid = lgb.Dataset(val[predictors].values, label=val['is_attributed'].values,
                              feature_name=predictors,
                              categorical_feature=categorical
                              )
        del val
        gc.collect()

        evals_results = {}

        if use_pse == True:
            file_path = './model/'+'pse_'+ str(model_type) +'_'+str(feature_type)  +'_'+str(data_type) + '.hdf5'
        else :
            file_path = './model/'+str(model_type) +'_'+str(feature_type)  +'_'+str(data_type) + '.hdf5'

        if os.path.exists(file_path):
            my_model = file_path
        else:
            my_model = None
        model = lgb.train(params,
                         dtrain,
                         valid_sets=[dtrain, dvalid],
                         valid_names=['train','valid'],
                         evals_result=evals_results,
                         num_boost_round=1000,
                         early_stopping_rounds=50,
                         # num_boost_round=5,
                         # early_stopping_rounds=1,
                         verbose_eval=True,
                         init_model = my_model,
                         feval=None)
        model.save_model(file_path)

        pred = model.predict(test[predictors], num_iteration=model.best_iteration)

    else:
        pred = np.zeros( shape=(len(test), 1) )

        folds = StratifiedShuffleSplit(n_splits = splits, test_size = 0.05, random_state = 182)

        class_pred = np.zeros(len(train))

        for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train[predictors], train['is_attributed'])):
            print ("goto %d fold :" % n_fold)
            if use_pse == True:
                X_train_n = train[predictors].iloc[trn_idx]
                X_train_n = pd.concat([X_train_n, pseudo[predictors]], axis=0)
                X_train_n = X_train_n.values
                Y_train_n = train['is_attributed'].iloc[trn_idx]
                Y_train_n = pd.concat([Y_train_n,pseudo['is_attributed']], axis=0)
                Y_train_n = Y_train_n.values
            else:
                X_train_n = train[predictors].iloc[trn_idx].values
                Y_train_n = train['is_attributed'].iloc[trn_idx].values
            X_valid_n = train[predictors].iloc[val_idx].values
            Y_valid_n = train['is_attributed'].iloc[val_idx].values
            dtrain = lgb.Dataset(X_train_n, label=Y_train_n,
                              feature_name=predictors,
                              categorical_feature=categorical
                              )

            dvalid = lgb.Dataset(X_valid_n, label=Y_valid_n,
                              feature_name=predictors,
                              categorical_feature=categorical
                              )

            evals_results = {}
            if use_pse == True:
                file_path = './model/'+'pse_'+str(model_type) +'_'+str(feature_type)  +'_'+str(data_type) +'_'+str(n_fold) +'.hdf5'
            else :
                file_path = './model/'+str(model_type) +'_'+str(feature_type)  +'_'+str(data_type) +'_'+str(n_fold) +'.hdf5'

            if os.path.exists(file_path):
                my_model = file_path
            else:
                my_model = None
            model = lgb.train(params, dtrain, valid_sets=[dtrain, dvalid], valid_names=['train','valid'],
                         evals_result=evals_results, num_boost_round=1000, early_stopping_rounds=50,
                         init_model = my_model,
                         verbose_eval=True, feval=None)

            model.save_model(file_path)
            class_pred[val_idx] = model.predict(X_valid_n, num_iteration=model.best_iteration)
            score = roc_auc_score(Y_valid_n, class_pred[val_idx])
            print("\t Fold %d : %.6f in %3d rounds" % (n_fold + 1, score, model.best_iteration))

            if n_fold > 0:
                pred = model.predict(test[predictors], num_iteration=model.best_iteration) + pred
            else:
                pred = model.predict(test[predictors], num_iteration=model.best_iteration)

            del X_valid_n,Y_train_n,X_train_n,Y_valid_n,dtrain, dvalid
            gc.collect()


        # class_pred = pd.DataFrame(class_pred)
        # oof_names = ['is_attributed_oof']
        # class_pred.columns = oof_names
        # print("Full roc auc scores : %.6f" % roc_auc_score(train['is_attributed'], class_pred[oof_names]))

        # Save OOF predictions - may be interesting for stacking...
        # file_name = 'oof/'+str(model_type) + '_' + str(feature_type) +'_' + str(data_type) + '_oof.csv'
        # class_pred.to_csv(file_name, index=False, float_format="%.6f")


    pred = pred / splits
    pred =pd.DataFrame(pred)
    pred.columns = target
    return pred

def m_xgb_model(train, test, model_type,feature_type,  data_type):

    if feature_type == 'andy_org':
        predictors = ['ip', 'device', 'app', 'os', 'channel', 'hour', 'n_channels', 'ip_app_count', 'ip_app_os_count']
    elif feature_type == 'andy_doufu':
        predictors = ['ip', 'device', 'app', 'os', 'channel', 'hour', 'n_channels', 'ip_app_count', 'ip_app_os_count', 'app_channel_count']
    categorical = ['ip', 'app', 'device', 'os', 'channel', 'hour']

    target = ['is_attributed']
    Y = train[target]
    train = train[predictors]
    test = test[predictors]

    splits = 3
    params = {'eta': 0.3,
          'tree_method': "gpu_hist",
          'grow_policy': "lossguide",
          'max_leaves': 1400,
          'max_depth': 0,
          'subsample': 0.9,
          'colsample_bytree': 0.7,
          'colsample_bylevel':0.7,
          'min_child_weight':0,
          'alpha':4,
          'objective': 'binary:logistic',
          'scale_pos_weight':99,
          'eval_metric': 'auc',
          'nthread':8,
          'random_state': 99,
            'gpu_id': 0,
            'max_bin': 16,
            'tree_method':'gpu_hist',
          'silent': True}
    # params = {'eta': 0.3,
    #       'grow_policy': "lossguide",
    #       'objective': 'binary:logistic',

    #       'max_depth' : 4,
    #       'gamma' : 9.9407,
    #       'eta' : 0.6223,
    #       'subsample' : 0.6875,
    #       'colsample_bytree' : 0.6915,
    #       'min_child_weight' : 2.7958,
    #       'max_delta_step' : 3,
    #       'seed' : 1001,
    #       'scale_pos_weight':99, # 40000000 : 480000
    #       'reg_alpha':0.0823, # default 0
    #       'reg_lambda':1.3776, # default 1

    #       'eval_metric': 'auc',
    #       'random_state': 99,
    #         'gpu_id': 0,
    #         'max_bin': 16,
    #         'tree_method':'gpu_hist',
    #       'silent': True}
    one_fold = False
    if one_fold == True:
        splits = 1
        X_train, X_valid, Y_train, Y_valid = train_test_split(train, Y, test_size=0.05, random_state=99)
        dtrain = xgb.DMatrix(X_train, Y_train)
        dvalid = xgb.DMatrix(X_valid, Y_valid)
        del X_train, Y_train, X_valid, Y_valid
        gc.collect()

        dtest = xgb.DMatrix(test)

        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
        file_path = './model/'+str(model_type) +'_'+str(feature_type)  +'_'+str(data_type) +'.hdf5'
        if os.path.exists(file_path):
            my_model = file_path
        else:
            my_model = None
        # model = xgb.train(params, dtrain, 1000, watchlist, maximize=True, early_stopping_rounds = 50, verbose_eval=5)
        model = xgb.train(params, dtrain, 5, watchlist, maximize=True, xgb_model=my_model,
            early_stopping_rounds = 1, verbose_eval=5)

        model.save_model(file_path)
        pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)

    else:
        pred = np.zeros( shape=(len(test), 1) )

        folds = StratifiedShuffleSplit(n_splits = splits, test_size = 0.005, random_state = 182)

        dtest = xgb.DMatrix(test)
        class_pred = np.zeros(len(train))
        for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train, Y)):
            print ("goto %d fold :" % n_fold)
            X_train_n = train[predictors].iloc[trn_idx].values
            Y_train_n = Y.iloc[trn_idx].values
            X_valid_n = train[predictors].iloc[val_idx].values
            Y_valid_n = Y.iloc[val_idx].values
            dtrain = xgb.DMatrix(X_train_n, Y_train_n)
            dvalid = xgb.DMatrix(X_valid_n, Y_valid_n)

            watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

            file_path = './model/'+str(model_type) +'_'+str(feature_type)  +'_'+str(data_type) +'_'+str(n_fold) +'.hdf5'
            if os.path.exists(file_path):
                my_model = file_path
            else:
                my_model = None
            model = xgb.train(params, dtrain, 1000, watchlist, maximize=True, xgb_model=my_model,
                early_stopping_rounds = 50, verbose_eval=1, )

            if n_fold > 0:
                pred = model.predict(dtest, ntree_limit=model.best_ntree_limit) + pred
            else:
                pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)

            class_pred[val_idx] = model.predict(xgb.DMatrix(X_valid_n), ntree_limit=model.best_ntree_limit)
            score = roc_auc_score(Y.iloc[val_idx], class_pred[val_idx])
            print("\t Fold %d : %.6f in %3d rounds" % (n_fold + 1, score, model.best_iteration))
            model.save_model(file_path)

            if n_fold > 0:
                pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)
            else :
                pred = pred + model.predict(dtest, ntree_limit=model.best_ntree_limit)

            del X_train_n,Y_train_n,X_valid_n, Y_valid_n, dtrain, dvalid
            gc.collect()

        class_pred = pd.DataFrame(class_pred)
        oof_names = ['is_attributed_oof']
        class_pred.columns = oof_names
        print("Full roc auc scores : %.6f" % roc_auc_score(Y, class_pred[oof_names]))

        # Save OOF predictions - may be interesting for stacking...
        # file_name = 'oof/'+str(model_type) + '_' + str(feature_type) +'_' + str(data_type) + '_oof.csv'
        # class_pred.to_csv(file_name, index=False, float_format="%.6f")


    pred = pred / splits
    pred =pd.DataFrame(pred)
    pred.columns = target

    return pred

def m_nn_model(x_train, y_train, x_valid, y_valid,test_df,model_type, feature_type, data_type,  file_path):

    features = x_train.columns
    print (features)

    emb_n = 50
    dense_n = 1000
    batch_size = 50000
    # batch_size = 20000
    epochs = 2
    lr_init, lr_fin = 0.001, 0.0001
    dr = 0.2
    lr = 0.001

    check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
                                  save_best_only = True, mode = "min")
    early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 5)

    emb_list = []
    row_list = []
    input_list = []
    m_ish = 0
    nn = 0

    # for n, feature in enumerate(features):
    #     if type(train[str(feature)][0]) != type(np.float16(1.0)):
    #         m +=1
    if feature_type == 'nano':
        emb_feature =  ['app','device','os', 'channel', 'hour',
              'nip_day_test_hh',  'nip_hh_os', 'nip_hh_dev']

        for n, feature in enumerate(emb_feature):
            input_list.append(Input(shape=[1], name = str(feature)))
            max_num = np.max([x_train[str(feature)].max(), test_df[str(feature)].max()])+1
            emb_list.append(Embedding(max_num, emb_n)(input_list[n]))

        # other_feature = list(set(features) - set(emb_feature) )
        input_list.append( Input(shape=[1], name = str('ip_app_nextClick')) )
        # emb_list.append(input_list[-1])


    # for n, feature in enumerate(features):
    #     if type(x_train[str(feature)][0]) != type(np.float16(1.0)):
    #         input_list.append(Input(shape=(1,1,), name = str(feature)))
    #         max_num = np.max([x_train[str(feature)].max(), test_df[str(feature)].max()])+1
    #         # emb_list.append(Embedding(max_num, emb_n)(input_list[n]))
    #         nn += 1
    #     else:
    #         input_list.append(Input(shape=(1,1,), name = str(feature)))
    #         m_ish += 1

    # input_list.append(Input(shape=(1,len(features)), name = str('all_feature')))
    # fe = concatenate(emb_list)

    fe = concatenate(emb_list)
    # fe = Input(shape=(1,len(features)), name = str('all_feature'))

    ############################
    # Old version
    ############################
    # s_dout = SpatialDropout1D(0.2)(fe)
    # fl = Flatten()(s_dout)
    # x = Dropout(dr)(Dense(dense_n,activation='relu')(fl))
    # x = Dropout(dr)(Dense(dense_n,activation='relu')(x))
    # gl = MaxPooling1D(pool_size=1, strides=1)(s_dout)
    # fl = Flatten()(gl)
    # x = concatenate([(x), (fl)])
    # outp = Dense(1,activation='sigmoid')(x)
    # model = Model(inputs=input_list, outputs=outp)

    ############################
    # New version 20180402
    ############################
    s_dout = SpatialDropout1D(0.2)(fe)
    fl1 = Flatten()(s_dout)
    conv = Conv1D(100, kernel_size=4, strides=1, padding='same')(s_dout)
    fl2 = Flatten()(conv)
    fl3 = input_list[-1]
    concat = concatenate([(fl1), (fl2), (fl3)])
    x = Dropout(dr)(Dense(dense_n,activation='relu')(concat))
    x = Dropout(dr)(Dense(dense_n,activation='relu')(x))
    outp = Dense(1,activation='sigmoid')(x)
    model = Model(inputs=input_list, outputs=outp)


    exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
    steps = int(len(x_train) / batch_size) * epochs
    lr_decay = exp_decay(lr_init, lr_fin, steps)
    optimizer_adam = Adam(lr=lr, decay=lr_decay)
    model.compile(loss='binary_crossentropy',optimizer=optimizer_adam,metrics=['accuracy'])

    print (model.summary())
    with timer("h_get_keras_data for train"):
        x_train = h_get_keras_data(x_train)
    with timer("h_get_keras_data for valid"):
        x_valid = h_get_keras_data(x_valid)

    ra_val = RocAucEvaluation(validation_data=(x_valid, y_valid), interval = 1)
    class_weight = {0:.01,1:.99} # magic
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, class_weight=class_weight,
        shuffle=True, verbose=1, validation_data = (x_valid, y_valid),
        callbacks = [ra_val, check_point, early_stop])

    return model


""""""""""""""""""""""""""""""
# Stacking
""""""""""""""""""""""""""""""
def app_tune_stack():
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    class_names_oof = []
    for c in class_names:
        class_names_oof.append(c+'_oof')

    sub = pd.read_csv('./input/sample_submission.csv')

    train_list, test_list =  h_get_train_test_list()
    num_file = len(train_list)

    train = h_prepare_data_train(train_list)
    test = h_prepare_data_test(test_list)
    train_target = train[class_names]

    param_test1 = {
        'max_depth': [3,4, 5],
        'num_leaves': [i for i in range(8,16, 2)]
    }
    param_test2 = {
        'learning_rate': [0.04, 0.05, 0.06,0.07,0.08,0.09, 0.1, 0.12],
    }
    param_test3 = {
        'colsample_bytree': [i / 100.0 for i in range(20, 100, 5)]
    }
    param_test4 = {
        'reg_lambda': [ i / 10 for i in range(0, 10)]
    }
    param_set = [
        param_test1,
        param_test2,
        param_test3,
        param_test4,
    ]

    param_dict_org= {
        "objective": "binary",
        "metric": {'auc'},
        "boosting_type": "gbdt",
        "num_threads": 4,

        "num_leaves": 10,
        "max_depth": 3,

        "colsample_bytree": .45,
        "min_data_in_leaf":24,
        "min_sum_hessian_in_leaf":.001,
        "learning_rate": 0.1,

        "bagging_fraction": .1,
        "bagging_freq":0,

        "reg_alpha": .8,
        "reg_lambda": .2,

        "max_bin": 24,
        "min_split_gain":.3,
        "verbose": -1,
        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
    }
    train_r = train.drop(class_names,axis=1)

    # X_train, X_valid, Y_train, Y_valid = train_test_split(train_r, train_target, test_size = 0.1, random_state=1982)
    X_train = train_r
    Y_train = train_target
    with timer ("Serching for best "):
        for class_name in class_names:
            param_dict = param_dict_org
            print (param_dict)
            bscores = []
            for param in param_set:
                with timer("goto serching ... ... "):
                    score , best_param = h_tuning_lgb(X_train, Y_train[class_name],param_dict, param)

                # time.sleep(5)
                print (type(best_param))
                for key in param_dict:
                    for key2 in best_param:
                        if key == key2:
                            param_dict[key] = best_param[key2]
                            print ("change %s to %f" % (key, best_param[key2]))

                one_scroe = {'score':score, 'param':param_dict}
                bscores.append(one_scroe)
            pfile = 'param_'+str(class_name) + '.csv'
            with open(pfile, 'w') as f:
                w = csv.DictWriter(f, param_dict.keys())
                w.writeheader()
                w.writerow(param_dict)

    print (param_dict)

    return

def app_stack():
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    class_names_oof = []
    for c in class_names:
        class_names_oof.append(c+'_oof')

    sub = pd.read_csv('./input/sample_submission.csv')

    train_list, test_list =  h_get_train_test_list()
    num_file = len(train_list)

    # stacker = LogisticRegression()
    # stacker = xgb.XGBClassifier()
    # stacker = lgb.LGBMClassifier(max_depth=4, metric="auc", n_estimators=125, num_leaves=9, boosting_type="gbdt",
    #                              learning_rate=0.1,  colsample_bytree=0.41,reg_lambda=0.9,
    #                         device = 'gpu',
    #                         gpu_platform_id=0,
    #                         gpu_device_id = 0,)

    X_train, X_valid, Y_train, Y_valid = train_test_split(train_r, train_target, test_size = 0.1, random_state=1982)
    # Fit and submit

    return

""""""""""""""""""""""""""""""
# Train
""""""""""""""""""""""""""""""
def h_tuning_xgb(train, train_target,tune_dict, param_test):

    param_dict = {
       'learning_rate' : 0.1,
       'n_estimators'  : 1000,
       'max_depth' : 5,
       'min_child_weight':1,
       'gamma':0,
       'subsample':0.8,
       'colsample_bytree':0.8,
       'scale_pos_weight':1,
       'reg_alpha':0,
       'reg_lambda':0,
       'booster':'gbtree', # 'gbtree','gblinear', 'dart'

    }
    param_dict = tune_dict

    gsearch = GridSearchCV(estimator=XGBClassifier(learning_rate=param_dict['learning_rate'],
                                                    n_estimators=param_dict['n_estimators'],
                                                    max_depth=param_dict['max_depth'],
                                                    min_child_weight=param_dict['min_child_weight'],
                                                    gamma=param_dict['gamma'],
                                                    subsample=param_dict['subsample'],
                                                    colsample_bytree=param_dict['colsample_bytree'],
                                                    scale_pos_weight=param_dict['scale_pos_weight'],
                                                    reg_alpha=param_dict['reg_alpha'],
                                                    reg_lambda=param_dict['reg_lambda'],
                                                    # gpu_id=0,
                                                    # max_bin = 16,
                                                    # tree_method = 'gpu_hist',
                                                    tree_method='hist',
                                                    objective='binary:logistic',
                                                    nthread=4,
                                                    seed=27),
                                                    param_grid=param_test, scoring='roc_auc', n_jobs=4, iid=False, cv=5, verbose=1)

    with timer("goto serch max_depth and min_child_wight"):
        gsearch.fit(train, train_target)
        print (gsearch.grid_scores_ )
        print (gsearch.best_params_ )
        print (gsearch.best_score_)
        return gsearch.best_score_, gsearch.best_params_


def h_tuning_lgb(train, train_target,tune_dict, param_test):

    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    params = tune_dict
    gsearch = GridSearchCV(estimator=lgb.LGBMClassifier(boosting_type="gbdt", objective="binary", metric="auc",
                            # num_threads = 4,

                            num_leaves = params["num_leaves"],
                            max_depth =  params["max_depth"],
                            n_estimators=125,
                            learning_rate =params["learning_rate"],
                            colsample_bytree=params["colsample_bytree"],
                            reg_lambda=params["reg_lambda"],
                            # device = 'gpu',
                            # gpu_platform_id=0,
                            # gpu_device_id = 0,
                            ) ,
                            param_grid=param_test, scoring='roc_auc', n_jobs=4, iid=False, cv=5, verbose=0)

    with timer("goto tuning lgb_wight"):
        gsearch.fit(train, train_target)
        print (gsearch.grid_scores_ )
        print (gsearch.best_params_ )
        print (gsearch.best_score_)

    return gsearch.best_score_, gsearch.best_params_

# Comment out any parameter you don't want to test

def XGB_CV(
          max_depth,
          max_leaves,
          # gamma,
          # min_child_weight,
          # max_delta_step,
          # subsample,
          # colsample_bytree,
          # reg_alpha,
          # reg_lambda,
          # eta,
         ):
      # 'learning_rate' : 0.1,
      #  'n_estimators'  : 1000,
      #  'max_depth' : 5,
      #  'min_child_weight':1,
      #  'gamma':0,
      #  'subsample':0.8,
      #  'colsample_bytree':0.8,
      #  'scale_pos_weight':1,
      #  'reg_alpha':0,
      #  'reg_lambda':0,
      #  'booster':'gbtree', # 'gbtree','gblinear', 'dart'

    global AUCbest
    global ITERbest

#
# Define all XGboost parameters
#
    # from joyo
    #       'subsample': 0.9,
    #       'colsample_bytree': 0.7,
    #       'colsample_bylevel':0.7,
    #       'min_child_weight':0,
    #       'alpha':4,
    paramt = {
              'booster' : 'gbtree',
              'max_depth' : int(max_depth),
              # 'gamma' : gamma,
              # 'eta' : float(eta),
              'eta' : .3,
              'objective' : 'binary:logistic',
              'alpha':4,
              'nthread' : 4,
              'silent' : True,
              'eval_metric': 'auc',
              'max_leaves':int(max_leaves),
              # 'subsample' : max(min(subsample, 1), 0),
              'subsample' :.9,
              'colsample_bylevel':0.7,
              # 'colsample_bytree' : max(min(colsample_bytree, 1), 0),
              'colsample_bytree' : .7,
              'min_child_weight' : 0,
              # 'min_child_weight' : min_child_weight,
              # 'max_delta_step' : int(max_delta_step),
              'seed' : 1001,
              'scale_pos_weight':9, # 40000000 : 480000
              # 'reg_alpha':float(reg_alpha), # default 0
              # 'reg_lambda':float(reg_lambda), # default 1
              'gpu_id': 0,
              'max_bin':16,
              'tree_method':'gpu_hist',
              'grow_policy': "lossguide",
              # 'tree_method':'hist',
              }

    folds = 5
    cv_score = 0

    print("\n Search parameters (%d-fold validation):\n %s" % (folds, paramt), file=log_file )
    log_file.flush()

    xgbc = xgb.cv(
                    paramt,
                    dtrain,
                    num_boost_round = 20000,
                    stratified = True,
                    nfold = folds,
#                    verbose_eval = 10,
                    early_stopping_rounds = 100,
                    metrics = 'auc',
                    show_stdv = True
               )

# This line would have been on top of this section
#    with capture() as result:

# After xgb.cv is done, this section puts its output into log file. Train and validation scores
# are also extracted in this section. Note the "diff" part in the printout below, which is the
# difference between the two scores. Large diff values may indicate that a particular set of
# parameters is overfitting, especially if you check the CV portion of it in the log file and find
# out that train scores were improving much faster than validation scores.

#    print('', file=log_file)
#    for line in result[1]:
#        print(line, file=log_file)
#    log_file.flush()

    val_score = xgbc['test-auc-mean'].iloc[-1]
    train_score = xgbc['train-auc-mean'].iloc[-1]
    print(' Stopped after %d iterations with train-auc = %f val-auc = %f ( diff = %f ) train-gini = %f val-gini = %f' % ( len(xgbc), train_score, val_score, (train_score - val_score), (train_score*2-1),
(val_score*2-1)) )
    if ( val_score > AUCbest ):
        AUCbest = val_score
        ITERbest = len(xgbc)

    return (val_score*2) - 1

def app_tune_xgb(train, feature_type):

    if feature_type == 'andy_org':
        predictors = ['ip', 'device', 'app', 'os', 'channel', 'hour', 'n_channels', 'ip_app_count', 'ip_app_os_count']
    elif feature_type == 'andy_doufu':
        predictors = ['ip', 'device', 'app', 'os', 'channel', 'hour', 'n_channels', 'ip_app_count', 'ip_app_os_count', 'app_channel_count']

    target = 'is_attributed'
    train_target = train[target]
    train = train[predictors]

    param_dict_org = {
       'learning_rate' : 0.1,
       'n_estimators'  : 1000,
       'max_depth' : 5,
       'min_child_weight':1,
       'gamma':0,
       'subsample':0.8,
       'colsample_bytree':0.8,
       'scale_pos_weight':1,
       'reg_alpha':0,
       'reg_lambda':0,
       'booster':'gbtree', # 'gbtree','gblinear', 'dart'
    }

    param_test1 = {
        'max_depth': [3,10, 2],
        'min_child_weight': [i for i in range(1,6, 2)]
    }
    param_test2 = {
        'gamma': [i / 10 for i in range(0, 10)],
    }
    param_test3 = {
        'colsample_bytree': [i / 100.0 for i in range(20, 100, 5)],
        'subsample':[i/10.0 for i in range(6,10)]
    }
    param_test4 = {
        'reg_alptha': [ i / 10 for i in range(0, 10)]
    }
    param_test5 = {
        'reg_lambda': [ i / 10 for i in range(0, 10)]
    }
    param_set = [
        param_test1,
        param_test2,
        param_test3,
        param_test4,
        param_test5,
    ]


    with timer ("Serching for best "):
        param_dict = param_dict_org
        print (param_dict)
        bscores = []
        for param in param_set:
            with timer("goto serching ... ... "):
                score , best_param = h_tuning_xgb(train, train_target, param_dict, param)

            # time.sleep(5)
            print (type(best_param))
            for key in param_dict:
                for key2 in best_param:
                    if key == key2:
                        param_dict[key] = best_param[key2]
                        print ("change %s to %f" % (key, best_param[key2]))

            one_scroe = {'score':score, 'param':param_dict}
            bscores.append(one_scroe)
        pfile = 'param_'+str(feature_type) + '.csv'
        with open(pfile, 'w') as f:
            w = csv.DictWriter(f, param_dict.keys())
            w.writeheader()
            w.writerow(param_dict)

    print (param_dict)

    return


def app_train(train, test, model_type,feature_type, data_type,use_pse, pseudo):

    with timer("goto train..."):
        if model_type == 'lgb':
            pred = m_lgb_model(train, test, model_type, feature_type, data_type,use_pse, pseudo)
        elif model_type == 'xgb':
            pred = m_xgb_model(train, test, model_type, feature_type, data_type)
    return pred

def app_tune_xgb_bayesian(train, feature_type):

    XGB_BO = BayesianOptimization(XGB_CV, {
                                     # 'max_depth': (2, 12),
                                     'max_depth': (0,5),
                                     'max_leaves': (0,2000)
                                     # 'gamma': (0.001, 10.0),
                                     # 'min_child_weight': (0, 20),
                                     # 'max_delta_step': (0, 10),
                                     # 'subsample': (0.4, 1.0),
                                     # 'colsample_bytree' :(0.4, 1.0),
                                     # 'reg_alpha' :(0, 1.0),
                                     # 'reg_lambda' :(0.1, 1.5),
                                     # 'eta' :(0.1, 1.0)
                                    })
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        XGB_BO.maximize(init_points=2, n_iter=3, acq='ei', xi=0.0)

    print('-'*130)
    print('Final Results')
    print('Maximum XGBOOST value: %f' % XGB_BO.res['max']['max_val'])
    print('Best XGBOOST parameters: ', XGB_BO.res['max']['max_params'])
    print('-'*130, file=log_file)
    print('Final Result:', file=log_file)
    print('Maximum XGBOOST value: %f' % XGB_BO.res['max']['max_val'], file=log_file)
    print('Best XGBOOST parameters: ', XGB_BO.res['max']['max_params'], file=log_file)
    log_file.flush()
    log_file.close()

    history_df = pd.DataFrame(XGB_BO.res['all']['params'])
    history_df2 = pd.DataFrame(XGB_BO.res['all']['values'])
    history_df = pd.concat((history_df, history_df2), axis=1)
    history_df.rename(columns = { 0 : 'gini'}, inplace=True)
    history_df['AUC'] = ( history_df['gini'] + 1 ) / 2
    history_df.to_csv('bayesian_xgb_grid.csv')

    return

def app_train_nn(train, test, model_type, feature_type, data_type):

    target = ['is_attributed']
    if feature_type == 'andy_org':
        feature_names = ['ip', 'device', 'app', 'os', 'channel', 'hour', 'n_channels', 'ip_app_count', 'ip_app_os_count']
    elif feature_type == 'andy_doufu':
        feature_names = ['ip', 'device', 'app', 'os', 'channel', 'hour', 'n_channels', 'ip_app_count', 'ip_app_os_count', 'app_channel_count']
    elif feature_type == 'pranav':
        feature_names = ['app','device','os', 'channel', 'hour', 'n_channels', 'ip_app_count', 'ip_app_os_count',
              'nip_day_test_hh', 'nip_day_hh', 'nip_hh_os', 'nip_hh_app', 'nip_hh_dev']
    elif feature_type == 'nano':
         cols = train.columns
         no_use = ['is_attributed', 'ip']
         feature_names = list(set(cols) - set(no_use))

    categorical = ['ip', 'app', 'device', 'os', 'channel', 'hour']

    splits = 1

    embed_size = 300

    m_batch_size = 32
    m_epochs = 4
    m_verbose = 1
    lr = 1e-3
    lr_d = 0
    units = 128
    dr = 0.2

    class_pred = np.ndarray(shape=(len(train), 1))

    with timer("Goto Train NN Model"):
        # folds = StratifiedKFold(n_splits=splits, shuffle=True, random_state=1)
        with timer("Goto StratifiedShuffleSplit ..."):
            folds = StratifiedShuffleSplit(n_splits = splits, test_size = 0.01, random_state = 182)

        for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train[feature_names], train[target])):

            print ("goto %d fold :" % n_fold)
            # print ("type(train[target]) is", type(train[target]))
            X_train_n = train[feature_names].iloc[trn_idx]
            Y_train_n = train[target].iloc[trn_idx].values
            X_valid_n = train[feature_names].iloc[val_idx]
            Y_valid_n = train[target].iloc[val_idx].values
            # print ("type(X_train_n) is", type(X_train_n))
            # print ("type(Y_train_n) is", type(Y_train_n))

            if model_type == 'nn': # nn
                if use_pse == True:
                    file_path = './model/'+'pse_'+str(model_type) +'_'+str(feature_type)  +'_'+str(data_type) + '.hdf5'
                else :
                    file_path = './model/'+str(model_type) +'_'+str(feature_type)  +'_'+str(data_type) + '.hdf5'
                # if os.path.exists(file_path):
                #     model = load_model(file_path)
                # else:
                model = m_nn_model(X_train_n, Y_train_n, X_valid_n, Y_valid_n,test,model_type, feature_type, data_type,  file_path)

        print("goto test")
        with timer("Goto prepare test Data"):
            test = h_get_keras_data(test)
        with timer("Goto predict test Data"):
            pred = model.predict(test)

        # with timer("Goto prepare oof Data"):
        #     oof_valid = h_get_keras_data(train[feature_names], feature_type)
        #     class_pred =pd.DataFrame(model.predict(oof_valid))

        # oof_names = ['is_attributed_oof']
        # class_pred.columns = oof_names
        # print("roc auc scores : %.6f" % roc_auc_score(train['is_attributed'], class_pred[oof_names]))

        # Save OOF predictions - may be interesting for stacking...
        # file_name = 'oof/'+str(model_type) + '_' + str(feature_type) +'_' + str(data_type) + '_oof.csv'
        # class_pred.to_csv(file_name, index=False, float_format="%.6f")

        pred = pred / splits
        pred =pd.DataFrame(pred)
        pred.columns = target

        return pred



""""""""""""""""""""""""""""""
# Ganerate Result
""""""""""""""""""""""""""""""
def g_make_single_submission(outfile, pred):
    submit = pd.read_csv('./input/test.csv', dtype='int', usecols=['click_id'])
    submit['is_attributed'] = pred
    submit.to_csv(outfile,float_format='%.3f', index=False)

def g_make_ooffile(outfile, pred, data_set):

    path_train ='./input/train.csv'
    path_test = './input/test.csv'
    train_cols = ['is_attributed']

    SKIP_ROWS = 100000000
    skip = range(1, SKIP_ROWS)

    dtypes = {
            'is_attributed' : 'uint8',
            }

    if data_set == 'set1':
        train = pd.read_csv(path_train, skiprows=skip, dtype=dtypes, header=0, usecols=train_cols)
    elif data_set == 'set0':
        train = pd.read_csv(path_train, nrows=SKIP_ROWS, dtype=dtypes, header=0, usecols=train_cols)
    elif data_set == 'setfull':
        train = pd.read_csv(path_train, dtype=dtypes, header=0, usecols=train_cols)
    elif data_set == 'set01':
        path_train ='./input/train_1.csv'
        train_1 = pd.read_csv(path_train, dtype=dtypes, header=0, usecols=train_cols)
        path_train ='./input/train_0.csv'
        train_0 = pd.read_csv(path_train, dtype=dtypes, header=0, usecols=train_cols)
        train = pd.concat([train_1, train_0])
        del train_0, train_1
        gc.collect()
    elif data_set == 'set001':
        path_train ='./input/train_1.csv'
        train_1 = pd.read_csv(path_train, dtype=dtypes, header=0, usecols=train_cols)
        path_train ='./input/train_00.csv'
        train_0 = pd.read_csv(path_train, dtype=dtypes, header=0, usecols=train_cols)
        train = pd.concat([train_1, train_0])
        del train_0, train_1
        gc.collect()
    elif data_set == 'set20':
        path_train ='./input/train_1.csv'
        train_1 = pd.read_csv(path_train, dtype=dtypes, header=0, usecols=train_cols)
        path_train ='./input/train_001.csv'
        train_0 = pd.read_csv(path_train, dtype=dtypes, header=0, usecols=train_cols)
        train = pd.concat([train_1, train_0])
        del train_0, train_1
        gc.collect()
    elif data_set == 'set21':
        path_train ='./input/train_1.csv'
        train_1 = pd.read_csv(path_train, dtype=dtypes, header=0, usecols=train_cols)
        path_train ='./input/train_002.csv'
        train_0 = pd.read_csv(path_train, dtype=dtypes, header=0, usecols=train_cols)
        train = pd.concat([train_1, train_0])
        del train_0, train_1
        gc.collect()

    train['is_attributed_oof'] = pred
    train.to_csv(outfile,float_format='%.3f', index=False)
    return


def g_make_pseudo_submission(outfile, m_pred):

    dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint16',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32'
            }

    submit = pd.read_csv('./input/test.csv', dtype=dtypes, header=0)
    submit['is_attributed'] = pred
    submit.to_csv(outfile,float_format='%.3f', index=False)


def h_tuning_bayesian():
    ##################################
    # use bayesian to find param for xgb
    ##################################
    if feature_type == 'andy_org':
        predictors = ['ip', 'device', 'app', 'os', 'channel', 'hour', 'n_channels', 'ip_app_count', 'ip_app_os_count']
    elif feature_type == 'andy_doufu':
        predictors = ['ip', 'device', 'app', 'os', 'channel', 'hour', 'n_channels', 'ip_app_count', 'ip_app_os_count', 'app_channel_count']
    categorical = ['ip', 'app', 'device', 'os', 'channel', 'hour']

    target = ['is_attributed']
    Y = train[target]
    train = train[predictors]

    dtrain = xgb.DMatrix(train, label=Y)

    app_tune_xgb_bayesian(train, feature_type)
    ##################################
    return

def my_simple_blend():
    # path_0 ='./output/set20lgbpranav977671_lb9694.csv'
    # path_1 ='./output/set20nnpranav_lb9671.csv'
    # path_2 ='./output/wordbatch_fm_ftrl.csv'
    # path_3 ='./output/set20lgb3foldpranav_lb9693.csv'

    path_0 ='./output/lgb_Usrnewness_9736.csv'
    path_1 ='./output/lgb_Usrnewness_9747.csv'
    path_2 ='./output/wordbatch_fm_ftrl_9752.csv'
    # path_3 ='./output/set20lgb3foldpranav_lb9693.csv'
    file0 = pd.read_csv(path_0)
    file1 = pd.read_csv(path_1)
    file2 = pd.read_csv(path_2)
    # file3 = pd.read_csv(path_3)
    pred = (file0['is_attributed'] + file1['is_attributed']+ file2['is_attributed']) /3
    outfile = 'output/blend_lgb9736_lgb9747_fm9752'+ '.csv'
    g_make_single_submission(outfile, pred)

def h_get_oof_file(data_type, model_type, feature_type, use_pse):

    train, test, pseudo = f_get_train_test_data(data_set, feature_type, use_pse)

    predictors = ['device', 'app', 'os', 'channel', 'hour', 'n_channels', 'ip_app_count', 'ip_app_os_count']
    categorical = ['app', 'device', 'os', 'channel', 'hour']
    if feature_type == 'andy_org':
        predictors = ['device', 'app', 'os', 'channel', 'hour', 'n_channels', 'ip_app_count', 'ip_app_os_count']
    elif feature_type == 'andy_doufu':
        predictors = ['device', 'app', 'os', 'channel', 'hour', 'n_channels', 'ip_app_count', 'ip_app_os_count', 'app_channel_count']
    elif feature_type == 'pranav':
        predictors = ['app','device','os', 'channel', 'hour', 'n_channels', 'ip_app_count', 'ip_app_os_count',
              'nip_day_test_hh', 'nip_day_hh', 'nip_hh_os', 'nip_hh_app', 'nip_hh_dev']

    if model_type == 'lgb':
        if use_pse == True:
            file_path = './model/'+str(model_type) +'_'+str(feature_type)  +'_'+str(data_type) + '.hdf5'
        else :
            file_path = './model/'+'pse_'+str(model_type) +'_'+str(feature_type)  +'_'+str(data_type) + '.hdf5'

        model = lgb.Booster(model_file=file_path)

        pred = model.predict(train[predictors], num_iteration=model.best_iteration)
    elif model_type == 'nn':
        if use_pse == True:
            file_path = './model/'+'pse_'+str(model_type) +'_'+str(feature_type)  +'_'+str(data_type) + '.hdf5'
        else :
            file_path = './model/'+str(model_type) +'_'+str(feature_type)  +'_'+str(data_type) + '.hdf5'

        with timer("h_get_keras_data for train"):
            x_train = h_get_keras_data(train[predictors], feature_type)

        model = load_model(file_path)
        pred = model.predict(x_train)

    outfile = 'oof/' + str(data_set) + str(model_type) + str(feature_type) + '_oof.csv'
    g_make_ooffile(outfile, pred,data_set)
    return

def s_stack():
    file_name = 'oof/set20lgbpranav_oof.csv'
    filei0 = pd.read_csv(file_name)

    file_name = 'oof/set20nnpranav_oof.csv'
    filei1 = pd.read_csv(file_name)

    target = ['is_attributed']
    target_oof = ['is_attributed_oof']

    train_target = filei0[target]

    filei0 = filei0[target_oof]
    filei0.columns = ['is_attributed_oof_0']

    filei1 = filei1[target_oof]
    filei1.columns = ['is_attributed_oof_1']

    file_name = 'output/set20lgbpranav977671_lb9694.csv'
    fileo0 = pd.read_csv(file_name)

    file_name = 'output/set20nnpranav_lb9671.csv'
    fileo1 = pd.read_csv(file_name)

    fileo0 = fileo0[target]
    fileo0.columns = ['is_attributed_oof_0']

    fileo1 = fileo1[target]
    fileo1.columns = ['is_attributed_oof_1']

    train = pd.concat ([filei0, filei1], axis = 1)
    test = pd.concat ([fileo0, fileo1], axis = 1)
    print (train.describe())
    print (test.describe())
    print (train_target.describe())
    model = LogisticRegression()
    model.fit(train, train_target)

    pred = model.predict_proba(test)[:,1]

    outfile = 'output/' + 'stack_' + '.csv'
    g_make_single_submission(outfile, pred)

    return


def app_stack_2():
    class_names = ['is_attributed']
    class_names_oof = []
    for c in class_names:
        class_names_oof.append(c+'_oof')

    train_list, test_list =  h_get_train_test_list()
    num_file = len(train_list)

    train = h_prepare_data_train(train_list)
    cols = train.columns
    cols = list(set(cols) - set(class_names))
    print (cols)
    test = h_prepare_data_test(test_list)
    # print (train.describe())
    # print (test.describe())

    train_target = train[class_names]
    # train.drop(['is_attributed'], axis=1)
    train = train[cols]
    print (train.describe())
    print (test.describe())

    # train_list, test_list =  h_get_train_test_list()
    # num_file = len(train_list)

    # train = h_prepare_data_train(train_list)
    # test = h_prepare_data_test(test_list)

    # stacker = LogisticRegression()
    # stacker = lgb.LGBMClassifier(max_depth=4, metric="auc", n_estimators=125, num_leaves=9, boosting_type="gbdt",
    #                              learning_rate=0.1,  colsample_bytree=0.41,reg_lambda=0.9,
    #                         device = 'gpu',
    #                         gpu_platform_id=0,
    #                         gpu_device_id = 0,)

    stacker = lgb.LGBMClassifier(metric="auc",boosting_type="gbdt",
                                 learning_rate=0.1,
                            scale_pos_weight=403,
                            device = 'gpu',
                            gpu_platform_id=0,
                            gpu_device_id = 0,)

    train_r = train

    # X_train, X_valid, Y_train, Y_valid = train_test_split(train_r, train_target, test_size = 0.1, random_state=1982)
    # Fit and submit
    X_train = train_r
    Y_train = train_target
    scores = []
    for label in class_names:
        print(label)
        score = cross_val_score(stacker, X_train, Y_train[label], cv=3, scoring='roc_auc',verbose=0)
        print("AUC:", score)
        scores.append(np.mean(score))
        stacker.fit(X_train, Y_train[label])
        pred = stacker.predict_proba(test)[:,1]
        # trn_pred = stacker.predict_proba(X_valid)[:,1]
        # print ("%s score : %f" % (str(label),  roc_auc_score(Y_valid[label], trn_pred)))
    print("CV score:", np.mean(scores))

    outfile = 'output/submission_stack_' + str(num_file) + 'file.csv'
    g_make_single_submission(outfile, pred)
    return


def f_get_nano_feature(data_set, feature_type):

    file_train = 'input/' + str(data_set)+'_'+ str(feature_type) + '_train.csv'
    file_test = 'input/' + str(data_set)+'_'+ str(feature_type) + '_test.csv'

    target_cols = ['is_attributed']
    train_cols = [
        'is_attributed'                    ,
        'app'                              ,
        'channel'                          ,
        'device'                           ,
        # 'ip'                               ,
        'os'                               ,
        'hour'                             ,
        'day'                              ,
        # 'minute'                           ,
        # 'second'                           ,
        'nip_day_test_hh'                  ,
        # 'ip_confRate'                      ,
        # 'app_confRate'                     ,
        # 'device_confRate'                  ,
        # 'os_confRate'                      ,
        # 'channel_confRate'                 ,
        # 'app_channel_confRate'             ,
        # 'app_os_confRate'                  ,
        # 'app_device_confRate'              ,
        'ip_app_channel_var_day'           ,
        'ip_app_os_var_hour'               ,
        'ip_day_channel_var_hour'          ,
        'ip_day_hour_count_channel'        ,
        'ip_app_count_channel'             ,
        'ip_app_os_count_channel'          ,
        'ip_app_day_hour_count_channel'    ,
        'ip_app_channel_mean_hour'         ,
        'app_AvgViewPerDistinct_ip'        ,
        'app_count_channel'                ,
        'channel_count_app'                ,
        'ip_nextClick'                     ,
        'ip_app_nextClick'                 ,
        'ip_channel_nextClick'             ,
        'ip_os_nextClick'                  ,
        'prev_identical_clicks'            ,
        'future_identical_clicks'          ,
        'prev_app_clicks'                  ,
        'future_app_clicks'                ,
        'nip_hh_os'                        ,
        'nip_hh_dev'                       ,
        ]
    test_cols = list(set(train_cols) -set(target_cols))
    dtypes = {
        'is_attributed'                    :'uint8',

        'app'                              :'uint16',
        'channel'                          :'uint16',
        'device'                           :'uint16',
        # 'ip'                               :'uint32',
        'os'                               :'uint16',
        'hour'                             :'uint8',
        'day'                              :'uint8',
        'minute'                           :'uint8',
        'second'                           :'uint8',
        'nip_day_test_hh'                  :'uint16',
        'ip_confRate'                      :'float16',
        'app_confRate'                     :'float16',
        'device_confRate'                  :'float16',
        'os_confRate'                      :'float16',
        'channel_confRate'                 :'float16',
        'app_channel_confRate'             :'float16',
        'app_os_confRate'                  :'float16',
        'app_device_confRate'              :'float16',
        'ip_app_channel_var_day'           :'float16',
        'ip_app_os_var_hour'               :'float16',
        'ip_day_channel_var_hour'          :'float16',
        'ip_day_hour_count_channel'        :'uint16',
        'ip_app_count_channel'             :'uint16',
        'ip_app_os_count_channel'          :'uint16',
        'ip_app_day_hour_count_channel'    :'uint16',
        'ip_app_channel_mean_hour'         :'float16',
        'app_AvgViewPerDistinct_ip'        :'float16',
        'app_count_channel'                :'uint32',
        'channel_count_app'                :'uint32',
        'ip_nextClick'                     :'float16',
        'ip_app_nextClick'                 :'float16',
        'ip_channel_nextClick'             :'float16',
        'ip_os_nextClick'                  :'float16',
        'prev_identical_clicks'            :'uint16',
        'future_identical_clicks'          :'uint16',
        'prev_app_clicks'                  :'uint32',
        'future_app_clicks'                :'uint32',
        'nip_hh_os'                        :'uint16',
        'nip_hh_dev'                       :'uint32',
    }

    with timer("goto open train"):
        train = pd.read_csv(file_train, dtype=dtypes, header=0, usecols=train_cols)
    # with timer("goto describe train"):
    #     print (train.describe(include='all'))
    # with timer("goto info train"):
    #     print (train.info())

    with timer("goto open test"):
        test = pd.read_csv(file_test, dtype=dtypes, header=0, usecols=test_cols)


    need_og = False
    if need_og == True:
        with timer('Binding the training and test set together...'):
            len_train = len(train)
            # print('The initial size of the train set is', len_train)
            train=train.append(test)

        gp = train[['ip', 'day', 'os', 'hour', 'channel']].groupby(by=['ip', 'os', 'day',
                 'hour'])[['channel']].count().reset_index().rename(index=str,
                 columns={'channel': 'nip_hh_os'})
        train = train.merge(gp, on=['ip','os','hour','day'], how='left')
        del gp
        print( "nip_hh_os max value = ", train.nip_hh_os.max() )
        train['nip_hh_os'] = train['nip_hh_os'].astype('uint16')
        gc.collect()

        gp = train[['ip', 'device', 'hour', 'day', 'channel']].groupby(by=['ip', 'device', 'day',
                 'hour'])[['channel']].count().reset_index().rename(index=str,
                 columns={'channel': 'nip_hh_dev'})
        train = train.merge(gp, on=['ip','device','day','hour'], how='left')
        del gp
        print( "nip_hh_dev max value = ", train.nip_hh_dev.max() )
        train['nip_hh_dev'] = train['nip_hh_dev'].astype('uint32')
        gc.collect()

        target = 'is_attributed'
        test = train[len_train:].copy().drop( target, axis=1 )
        train = train[:len_train]


    save_file = False
    if save_file == True:
        train.to_csv(file_train, index=False)
        test.to_csv(file_test, index=False)

    return train, test, None


""""""""""""""""""""""""""""""
# Main Func
""""""""""""""""""""""""""""""
AUCbest = -1.
ITERbest = 0

if __name__ == '__main__':
    # from andy :set0 set1 setfull
    # sample all 1 and random 0 :set01
    # sample all 1 and first part 0 :set001
    # sample all 1 and half (1/2sample) 0: set20 set21
    data_set = 'set20'
    model_type = 'nn' # xgb lgb nn
    # andy_org andy_doufu 'pranav' nano
    feature_type = 'nano' #
    use_pse = False

    # app_stack_2()
    # with timer("genarete oof file ..."):
    #     h_get_oof_file(data_set, model_type, feature_type, use_pse)
    # my_simple_blend()
    # h_get_pseudo_data()
    ##################################
    # traing for nn
    ##################################
    with timer ("get train, test , pseudo data ..."):
        train, test, pseudo = f_get_nano_feature(data_set, feature_type)
        # train, test, pseudo = f_get_train_test_data(data_set, feature_type, use_pse)
    print (data_set, model_type, feature_type, 'use pse :', str(use_pse) )
    print (train.info())
    print (test.info())
    if model_type == 'xgb' or model_type == 'lgb':
        print ("goto train ", str(model_type) )
        pred =  app_train(train, test, model_type,feature_type, data_set,use_pse, pseudo)
    elif model_type == 'nn':
        pred = app_train_nn(train, test, model_type, feature_type, data_set)

    outfile = 'output/' + str(data_set) + str(model_type) + str(feature_type) + '.csv'
    g_make_single_submission(outfile, pred)
    ##################################


    # outfile = 'pseudo/' + str(data_set) + str(model_type) + str(feature_type) + '_pseudo_test.csv'
    # g_make_pseudo_submission(outfile, pred)

    print('[{}] All Done!!!'.format(time.time() - start_time))


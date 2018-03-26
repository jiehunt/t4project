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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectFromModel

from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

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


""""""""""""""""""""""""""""""
# Feature
""""""""""""""""""""""""""""""

def f_get_train_test_data(data_set, feature_type):

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
    with timer('Loading the test data...'):
        test = pd.read_csv(path_test, dtype=dtypes, header=0, usecols=test_cols)

    with timer('Binding the training and test set together...'):
        len_train = len(train)
        print('The initial size of the train set is', len_train)
        train=train.append(test)


    with timer('Orgnizing Target Data...'):
        target = 'is_attributed'
        train.loc[train[target].isnull(),target] = 99
        train[target] = train[target].astype('uint8')
        print (train.info())
        del test
        gc.collect()


    with timer("Creating new time features: 'hour' and 'day'..."):
        train['hour'] = pd.to_datetime(train.click_time).dt.hour.astype('uint8')
        train['day'] = pd.to_datetime(train.click_time).dt.day.astype('uint8')

        train.drop( 'click_time', axis=1, inplace=True )
        gc.collect()

        # train['click_hour'] = pd.to_datetime(train.attributed_time).dt.hour.astype('uint8')
        # train['click_day'] = pd.to_datetime(train.attributed_time).dt.day.astype('uint8')

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

    test = train[len_train:].copy().drop( target, axis=1 )
    train = train[:len_train]
    print('The size of the test set is ', len(test))
    print('The size of the train set is ', len(train))

    return train, test



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

def m_lgb_model(train, test):

    if feature_type == 'andy_org':
        predictors = ['ip', 'device', 'app', 'os', 'channel', 'hour', 'n_channels', 'ip_app_count', 'ip_app_os_count']
    elif feature_type == 'andy_doufu':
        predictors = ['ip', 'device', 'app', 'os', 'channel', 'hour', 'n_channels', 'ip_app_count', 'ip_app_os_count', 'app_channel_count']
    categorical = ['ip', 'app', 'device', 'os', 'channel', 'hour']

    target = 'is_attributed'

    params = {
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
        'scale_pos_weight':99,
        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        }

    len_train = len(train)
    r = 0.1 # the fraction of the train data to be used for validation
    val = train[(len_train-round(r*len_train)):len_train]
    print('The size of the validation set is ', len(val))

    train = train[:(len_train-round(r*len_train))]
    print('The size of the train set is ', len(train))

    dtrain = lgb.Dataset(train[predictors].values, label=train[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
    del train
    gc.collect()

    dvalid = lgb.Dataset(val[predictors].values, label=val[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
    del val
    gc.collect()


    dvalid = lgb.Dataset(val[predictors].values, label=val[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
    del val
    gc.collect()

    evals_results = {}

    lgb_model = lgb.train(params,
                     dtrain,
                     valid_sets=[dtrain, dvalid],
                     valid_names=['train','valid'],
                     evals_result=evals_results,
                     num_boost_round=1000,
                     early_stopping_rounds=50,
                     verbose_eval=True,
                     feval=None)

    pred = lgb_model.predict(test[predictors], num_iteration=lgb_model.best_iteration)

    return pred

def m_xgb_model(train, test, feature_type):

    if feature_type == 'andy_org':
        predictors = ['ip', 'device', 'app', 'os', 'channel', 'hour', 'n_channels', 'ip_app_count', 'ip_app_os_count']
    elif feature_type == 'andy_doufu':
        predictors = ['ip', 'device', 'app', 'os', 'channel', 'hour', 'n_channels', 'ip_app_count', 'ip_app_os_count', 'app_channel_count']
    categorical = ['ip', 'app', 'device', 'os', 'channel', 'hour']

    target = 'is_attributed'
    Y = train[target]
    train = train[predictors]
    test = test[predictors]
    params = {'eta': 0.3,
          'tree_method': "hist",
          'grow_policy': "lossguide",
          'max_leaves': 1400,
          'max_depth': 0,
          'subsample': 0.9,
          'colsample_bytree': 0.7,
          'colsample_bylevel':0.7,
          'min_child_weight':0,
          'alpha':4,
          'objective': 'binary:logistic',
          'scale_pos_weight':9,
          'eval_metric': 'auc',
          'nthread':8,
          'random_state': 99,
            'gpu_id': 0,
            'max_bin': 16,
            'tree_method':'gpu_hist',
          'silent': True}
    X_train, X_valid, Y_train, Y_valid = train_test_split(train, Y, test_size=0.1, random_state=99)
    dtrain = xgb.DMatrix(X_train, Y_train)
    dvalid = xgb.DMatrix(X_valid, Y_valid)
    del X_train, Y_train, X_valid, Y_valid
    gc.collect()

    dtest = xgb.DMatrix(test)

    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    model = xgb.train(params, dtrain, 200, watchlist, maximize=True, early_stopping_rounds = 25, verbose_eval=5)

    pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)

    return pred

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
        # num_leaves         default=31, type=int, alias=num_leaf
        # max_depth          default=-1, type=int
        # feature_fraction   default=1.0, type=double, 0.0 < feature_fraction < 1.0 alias=sub_feature, colsample_bytree
        # min_data_in_leaf   default=20, type=int
        # min_sum_hessian_in_leaf    default=1e-3, type=double
        # learning_rate      default=0.1
        # max_bin            default=255, type=int
        # bagging_fraction   default=1.0, type=double, 0.0 < bagging_fraction < 1.0
        # bagging_freq       default=0
        # lambda_l1   default=0, type=double, alias=reg_alpha
        # lambda_l2   default=0, type=double, alias=reg_lambda
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
                                                    gpu_id=0,
                                                    max_bin = 16,
                                                    tree_method = 'gpu_hist',
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

def app_tune_xgb(train, test, feature_type):

    target = 'is_attributed'
    train_target = train[target]
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


def app_train(train, test, model_type,feature_type):

    with timer("goto train..."):
        if model_type == 'lgb':
            pred = m_lgb_model(train, test, feature_type)
        elif model_type == 'xgb':
            pred = m_xgb_model(train, test, feature_type)
    return pred

""""""""""""""""""""""""""""""
# Ganerate Result
""""""""""""""""""""""""""""""
def m_make_single_submission(outfile, m_pred):
    submit = pd.read_csv('./input/test.csv', dtype='int', usecols=['click_id'])
    submit['is_attributed'] = pred
    submit.to_csv(outfile,float_format='%.3f', index=False)

""""""""""""""""""""""""""""""
# Main Func
""""""""""""""""""""""""""""""

if __name__ == '__main__':

    data_set = 'set1' # set0 set1 setfull
    model_type = 'xgb' # xgb lgb
    feature_type = 'andy_org' # andy_org andy_doufu
    train, test = f_get_train_test_data(data_set, feature_type)

    print (train.info())
    print (test.info())
    # pred =  app_train(train, test, model_type,feature_type):


    app_tune_xgb(train, test,feature_type)

    # outfile = 'output/' + str(data_set) + str(model_type) + str(feature_type) + '.csv'
    # m_make_single_submission(outfile, pred)


    print('[{}] All Done!!!'.format(time.time() - start_time))


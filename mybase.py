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

def h_get_keras_data(dataset, feature_type):
    # if feature_type == 'andy_org':
    #     feature_names = ['ip', 'device', 'app', 'os', 'channel', 'hour', 'n_channels', 'ip_app_count', 'ip_app_os_count']
    # elif feature_type == 'andy_doufu':
    #     feature_names = ['ip', 'device', 'app', 'os', 'channel', 'hour', 'n_channels', 'ip_app_count', 'ip_app_os_count', 'app_channel_count']

    print (type(dataset))
    columns = dataset.columns

    X = {}
    for name in columns:
        X[str(name)] = np.array(dataset[[str(name)]])

    return X

""""""""""""""""""""""""""""""
# Feature
""""""""""""""""""""""""""""""

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

    with timer("Creating new time features: 'hour' and 'day'..."):
        train['hour'] = pd.to_datetime(train.click_time).dt.hour.astype('uint8')
        train['day'] = pd.to_datetime(train.click_time).dt.day.astype('uint8')
        train.drop( 'click_time', axis=1, inplace=True )
        gc.collect()

        # train['click_hour'] = pd.to_datetime(train.attributed_time).dt.hour.astype('uint8')
        # train['click_day'] = pd.to_datetime(train.attributed_time).dt.day.astype('uint8')

    if feature_type == 'pranav':
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
    if feature_type == 'andy_org':
        predictors = ['device', 'app', 'os', 'channel', 'hour', 'n_channels', 'ip_app_count', 'ip_app_os_count']
    elif feature_type == 'andy_doufu':
        predictors = ['device', 'app', 'os', 'channel', 'hour', 'n_channels', 'ip_app_count', 'ip_app_os_count', 'app_channel_count']
    elif feature_type == 'pranav':
        predictors = ['app','device','os', 'channel', 'hour', 'n_channels', 'ip_app_count', 'ip_app_os_count',
              'nip_day_test_hh', 'nip_day_hh', 'nip_hh_os', 'nip_hh_app', 'nip_hh_dev']

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
        'scale_pos_weight':99,
        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        }
    params = { ## get from andy_pranva
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric':'auc',
        # 'learning_rate': 0.3,
        'learning_rate': 0.1,
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
        'scale_pos_weight':99.7, # because training data is extremely unbalanced
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
            file_path = './model/'+str(model_type) +'_'+str(feature_type)  +'_'+str(data_type) + '.hdf5'
        else :
            file_path = './model/'+'pse_'+str(model_type) +'_'+str(feature_type)  +'_'+str(data_type) + '.hdf5'

        if os.path.exists(file_path):
            my_model = file_path
        else:
            my_model = None
        model = lgb.train(params,
                         dtrain,
                         valid_sets=[dtrain, dvalid],
                         valid_names=['train','valid'],
                         evals_result=evals_results,
                         num_boost_round=5000,
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
    input_list = []
    for n, feature in enumerate(features):
        max_num = np.max([x_train[str(feature)].max(), test_df[str(feature)].max()])+1
        input_list.append(Input(shape=[1], name = str(feature)))
        emb_list.append(Embedding(max_num, emb_n)(input_list[n]))

    fe = concatenate(emb_list)

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
    concat = concatenate([(fl1), (fl2)])
    x = Dropout(dr)(Dense(dense_n,activation='relu')(concat))
    x = Dropout(dr)(Dense(dense_n,activation='relu')(x))
    outp = Dense(1,activation='sigmoid')(x)
    model = Model(inputs=input_list, outputs=outp)


    exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
    steps = int(len(train) / batch_size) * epochs
    lr_decay = exp_decay(lr_init, lr_fin, steps)
    optimizer_adam = Adam(lr=lr, decay=lr_decay)
    model.compile(loss='binary_crossentropy',optimizer=optimizer_adam,metrics=['accuracy'])

    print (model.summary())
    with timer("h_get_keras_data for train"):
        x_train = h_get_keras_data(x_train, feature_type)
    with timer("h_get_keras_data for valid"):
        x_valid = h_get_keras_data(x_valid, feature_type)

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

    if feature_type == 'andy_org':
        feature_names = ['ip', 'device', 'app', 'os', 'channel', 'hour', 'n_channels', 'ip_app_count', 'ip_app_os_count']
    elif feature_type == 'andy_doufu':
        feature_names = ['ip', 'device', 'app', 'os', 'channel', 'hour', 'n_channels', 'ip_app_count', 'ip_app_os_count', 'app_channel_count']
    elif feature_type == 'pranav':
        feature_names = ['app','device','os', 'channel', 'hour', 'n_channels', 'ip_app_count', 'ip_app_os_count',
              'nip_day_test_hh', 'nip_day_hh', 'nip_hh_os', 'nip_hh_app', 'nip_hh_dev']
    categorical = ['ip', 'app', 'device', 'os', 'channel', 'hour']

    target = ['is_attributed']

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
            print ("type(train[target]) is", type(train[target]))
            X_train_n = train[feature_names].iloc[trn_idx]
            Y_train_n = train[target].iloc[trn_idx].values
            X_valid_n = train[feature_names].iloc[val_idx]
            Y_valid_n = train[target].iloc[val_idx].values
            print ("type(X_train_n) is", type(X_train_n))
            print ("type(Y_train_n) is", type(Y_train_n))

            if model_type == 'nn': # nn
                file_path = './model/'+str(model_type) +'_'+str(feature_type)  +'_'+str(data_type) + '.hdf5'
                # if os.path.exists(file_path):
                #     model = load_model(file_path)
                # else:
                model = m_nn_model(X_train_n, Y_train_n, X_valid_n, Y_valid_n,test,model_type, feature_type, data_type,  file_path)

        print("goto test")
        with timer("Goto prepare test Data"):
            test = h_get_keras_data(test, feature_type)
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

def g_make_ooffile(outfile, pred):

    train_cols = ['is_attributed']
    oof = pd.read_csv('./input/train.csv', dtype='uint8', usecols=train_cols)
    oof['is_attributed_oof'] = pred
    oof.to_csv(outfile,float_format='%.3f', index=False)


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
    path_0 ='./output/set20lgbpranav977671.csv'
    path_1 ='./output/set01nnandy_org.csv'
    path_2 ='./fun/set20lgbandy_org.csv'

    file0 = pd.read_csv(path_0)
    file1 = pd.read_csv(path_1)
    file2 = pd.read_csv(path_2)
    pred = (file0['is_attributed'] + file1['is_attributed']+ file2['is_attributed']) /3
    outfile = 'output/blend_set01nn_set001lgb_set20lgb9694_'+ str(feature_type) + '.csv'
    g_make_single_submission(outfile, pred)

def h_get_oof_file(data_type, model_type, feature_type, use_pse):

    train, test, pseudo = f_get_train_test_data(data_set, feature_type, use_pse)


    if use_pse == True:
        file_path = './model/'+str(model_type) +'_'+str(feature_type)  +'_'+str(data_type) + '.hdf5'
    else :
        file_path = './model/'+'pse_'+str(model_type) +'_'+str(feature_type)  +'_'+str(data_type) + '.hdf5'

    model = lgb.Booster(model_file=file_path)

    pred = model.predict(test[predictors], num_iteration=model.best_iteration)

    outfile = 'oof/' + str(data_set) + str(model_type) + str(feature_type) + '.csv'
    g_make_ooffile(outfile, pred)
    return
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
    model_type = 'lgb' # xgb lgb nn
    feature_type = 'pranav' # andy_org andy_doufu 'pranav'
    use_pse = False

    with timer("genarete oof file ..."):
        h_get_oof_file(data_set, model_type, feature_type, use_pse)
    # my_simple_blend()
    # h_get_pseudo_data()
    ##################################
    # traing for nn
    ##################################
    # train, test, pseudo = f_get_train_test_data(data_set, feature_type, use_pse)
    # print (data_set, model_type, feature_type, 'use pse :', str(use_pse) )
    # print (train.info())
    # print (test.info())
    # if model_type == 'xgb' or model_type == 'lgb':
    #     print ("goto train ", str(model_type) )
    #     pred =  app_train(train, test, model_type,feature_type, data_set,use_pse, pseudo)
    # elif model_type == 'nn':
    #     pred = app_train_nn(train, test, model_type, feature_type, data_set)

    # outfile = 'output/' + str(data_set) + str(model_type) + str(feature_type) + '.csv'
    # g_make_single_submission(outfile, pred)
    ##################################


    # outfile = 'pseudo/' + str(data_set) + str(model_type) + str(feature_type) + '_pseudo_test.csv'
    # g_make_pseudo_submission(outfile, pred)

    print('[{}] All Done!!!'.format(time.time() - start_time))


# sklearn:


# changing to sing path ONLY FOR JOB SUBMISSION
import pandas as pd

df = pd.read_pickle('asset_data_pick.pkl')

# import other relevant packages
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import pickle
import gzip

import numpy as np

import time
import functools

# tf.debugging.set_log_device_placement(True)
tf.config.set_visible_devices([], 'GPU')
pd.options.mode.chained_assignment = None
import concurrent.futures

# print(tf.config.list_physical_devices(), flush=True)
#
# y_dict = {}
# # read data
# for id in df['permno'].unique():
#     #     print(s)
#     #     temp=df['RET'].loc[df['permno']==id].shift(1)
#     y_dict.update(dict(zip(df['key'].loc[df['permno'] == id].values,
#                            df['RET'].loc[df['permno'] == id].shift(1).values)))
#
# df['RET'] = df['key'].map(y_dict)
#
# df.dropna(inplace=True)
# all_y_info = df.pop('y')
# cols_todrop = ['DATE', 'datetime', 'month', 'sic2_name']
# df.drop(columns=cols_todrop, inplace=True)
#
# # define t, permno dict
#
# time_dict = dict(zip(df.index.values, df['t'].values))
# permno_dict = dict(zip(df.index.values, df['permno'].values))
#
# df.dropna(inplace=True)
#
#
# # function defenition
# def time_series_convertor(seq_len, batch_size, df, y_variable, sequence_stride=1,
#                           sampling_rate=1):  # y_variable is string, df is a sequential only data indexed via time, step
#     y = df[y_variable]
#     dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
#         df[:-1],
#         y[seq_len:],
#         sequence_length=seq_len,
#         sequence_stride=sequence_stride,
#         sampling_rate=sampling_rate,
#         batch_size=batch_size,
#         shuffle=False,
#         seed=128,
#         start_index=None,
#     )
#     return dataset
#
#
# def main_task(df_train_scaled,
#               y_variable, seq_len, batch_size,
#               sequence_stride, sampling_rate, id):
#     i = 0
#     #     print(id)
#     time_data = df_train_scaled.loc[df_train_scaled['permno'] == id]
#     time_list = time_data['t'].unique()[seq_len:]
#     time_data.set_index('t', inplace=True)
#     time_data.drop(columns=['permno', 'key'], inplace=True)
#     if len(time_data) <= seq_len:
#         return None, None, None
#     seq_data = time_series_convertor(seq_len, batch_size, time_data, y_variable, sequence_stride=1, sampling_rate=1)
#     for b in seq_data:
#         x, y = b
#     cross_df = pd.DataFrame(columns=['t', 'permno', 'key', 'y_value'])
#     #         len_t=np.arange(min_t+seq_len,max_t+1)
#     cross_df['t'] = time_list
#     cross_df['y_value'] = y
#     cross_df = cross_df.assign(permno=id)
#     cross_df['key'] = cross_df['t'].astype(str) + "_" + cross_df['permno'].astype(str)
#     #     i=i+1
#     x = tf.cast(x, tf.float32)
#     y = tf.cast(y, tf.float32)
#     return x, y, cross_df
#
#
# # putting togther all the sequential data
# import time
#
# t1 = time.perf_counter()
# res = []
# y_variable = 'RET'
# seq_len = 24
# batch_size = 300000000
# sequence_stride = 1
# sampling_rate = 1
#
# partial_main = functools.partial(main_task, df,
#                                  y_variable, seq_len, batch_size,
#                                  sequence_stride, sampling_rate)
# firm_list = df['permno'].unique().tolist()  # set his to everyone for final test
#
# import concurrent.futures
#
# with concurrent.futures.ThreadPoolExecutor() as executor:
#     res = list(executor.map(partial_main, firm_list))
#
# t2 = time.perf_counter()
# print(f'finished paralel prepros in {t2 - t1} seconds', flush=True)
# print(f'Number of Thread workers:  {executor._max_workers} ', flush=True)



def merge_firm(res, year):
    i = 0
    train_end = (year * 12) - 1
    train_begin = train_end - 359

    valid_begin = train_end + 1
    valid_end = valid_begin + 12

    test_begin=valid_end+1
    test_end=test_begin+12

    for b in res:
        if b == (None, None, None):
            continue
        df = b[2]
        # train_bit
        y_info_train = df.loc[(df['t'] >= train_begin) & (df['t'] <= train_end)]
        if len(y_info_train) < 1:
            s_tr = len(b[0]) + 100
            e_tr = len(b[0]) + 101
        else:
            s_tr = y_info_train.index[0]
            e_tr = y_info_train.index[-1]
        x_train = b[0][s_tr:e_tr + 1]  # maybe e_tr+1
        y_train = b[1][s_tr:e_tr + 1]  # maybe e_tr+1
        #valid_bit
        y_info_valid=df.loc[(df['t'] >= valid_begin) & (df['t'] <= valid_end)]
        if len(y_info_valid) < 1:
            s_va = len(b[0]) + 100
            e_va = len(b[0]) + 101
        else:
            s_va = y_info_valid.index[0]
            e_va = y_info_valid.index[-1]
        x_valid = b[0][s_va:e_va + 1]  # maybe e_tr+1
        y_valid = b[1][s_va:e_va + 1]  # maybe e_tr+1


        # test_bit
        y_info_test = df.loc[(df['t'] >= test_begin) & (df['t'] <= test_end)]
        if len(y_info_test) < 1:
            s_te = len(b[0]) + 100
            e_te = len(b[0]) + 101
        else:
            s_te = y_info_test.index[0]
            e_te = y_info_test.index[-1]
        x_test = b[0][s_te:e_te + 1]  # maybe e_tr+1
        y_test = b[1][s_te:e_te + 1]  # maybe e_tr+1

        if i > 0:
            # concat data ids
            y_id_train_concat = pd.concat([y_id_train_concat, y_info_train], ignore_index=True)
            y_id_test_concat = pd.concat([y_id_test_concat, y_info_test], ignore_index=True)
            y_id_valid_concat = pd.concat([y_id_valid_concat, y_info_valid], ignore_index=True)
            # train
            concat_x_train = tf.concat([concat_x_train, x_train], 0)
            concat_y_train = tf.concat([concat_y_train, y_train], 0)
            # test
            concat_x_test = tf.concat([concat_x_test, x_test], 0)
            concat_y_test = tf.concat([concat_y_test, y_test], 0)

            #valid
            concat_x_valid = tf.concat([concat_x_valid, x_valid], 0)
            concat_y_valid = tf.concat([concat_y_valid, y_valid], 0)
        else:
            y_id_train_concat = y_info_train
            y_id_test_concat = y_info_test
            y_id_valid_concat = y_info_valid
            # train
            concat_x_train = x_train
            concat_y_train = y_train
            # test
            concat_x_test = x_test
            concat_y_test = y_test
            #valid
            # test
            concat_x_valid = x_valid
            concat_y_valid = y_valid
        i = i + 1
    return y_id_train_concat, y_id_test_concat, concat_x_train, concat_y_train, concat_x_test, concat_y_test, y_id_valid_concat, concat_x_valid,concat_y_valid
def save_zipped_pickle(obj, filename, protocol=-1):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)


def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object


res=load_zipped_pickle('res_lstm_all.pkl')
year_list = []
for year in range(30, 31):
    # defining the concept of time beg, end
    year_list.append(year)
    filename= 'merge_list_valid_inc'+ str(year)+'.pkl'
    t1 = time.perf_counter()
    merge_list=merge_firm(res, year)
    t2 = time.perf_counter()
    print(f'the merge  function in paralel ran in {t2 - t1} seconds', flush=True)
    save_zipped_pickle(merge_list, filename)
    print(filename, flush=True)

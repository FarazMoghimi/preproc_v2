#for this function to work, the y variable needs to be unlagged, the function will lag by it self
# something like this needs to be done if the data is already lagged:

y_dict = {}
# read data
for id in df['permno'].unique():
    #     print(s)
    #     temp=df['RET'].loc[df['permno']==id].shift(1)
    y_dict.update(dict(zip(df['key'].loc[df['permno'] == id].values,
                           df[y_variable].loc[df['permno'] == id].shift(1).values)))

df[y_variable] = df['key'].map(y_dict)


def time_series_convertor(seq_len, batch_size, df, y_variable, sequence_stride=1,
                          sampling_rate=1):  # y_variable is string, df is a sequential only data indexed via time, step
    y = df[y_variable]
    dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
        df[:-1],
        y[seq_len:],
        sequence_length=seq_len,
        sequence_stride=sequence_stride,
        sampling_rate=sampling_rate,
        batch_size=batch_size,
        shuffle=False,
        seed=128,
        start_index=None,
    )
    return dataset


def main_task(df_train_scaled,
              y_variable, seq_len, batch_size,
              sequence_stride, sampling_rate, id):
    i = 0
    #     print(id)
    time_data = df_train_scaled.loc[df_train_scaled['permno'] == id]
    time_list = time_data['t'].unique()[seq_len-1:-1]
    time_data.set_index('t', inplace=True)
    time_data.drop(columns=['permno', 'key'], inplace=True)
    if len(time_data) <= seq_len:
        return None, None, None
    seq_data = time_series_convertor(seq_len, batch_size, time_data, y_variable, sequence_stride=1, sampling_rate=1)
    for b in seq_data:
        x, y = b
    cross_df = pd.DataFrame(columns=['t', 'permno', 'key', 'y_value'])
    #         len_t=np.arange(min_t+seq_len,max_t+1)
    cross_df['t'] = time_list
    cross_df['y_value'] = y
    cross_df = cross_df.assign(permno=id)
    cross_df['key'] = cross_df['t'].astype(str) + "_" + cross_df['permno'].astype(str)
    #     i=i+1
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    return x, y, cross_df


# # for a single firm: 
# x, y, cross_df=main_task(firm,
#               'new_y', 24, 8000,
#               1, 1, 42550)

#from this point forward paralel threading is used to do this for all the firms (cros id) of the entire sample

import time

t1 = time.perf_counter()
res = []
seq_len = 24
batch_size = 300000000
sequence_stride = 1
sampling_rate = 1

partial_main = functools.partial(main_task, df,
                                 y_variable, seq_len, batch_size,
                                 sequence_stride, sampling_rate)
firm_list = df['permno'].unique().tolist()  # set his to everyone for final test

import concurrent.futures

with concurrent.futures.ThreadPoolExecutor() as executor:
    res = list(executor.map(partial_main, firm_list))

t2 = time.perf_counter()
print(f'finished paralel prepros in {t2 - t1} seconds', flush=True)
print(f'Number of Thread workers:  {executor._max_workers} ', flush=True)

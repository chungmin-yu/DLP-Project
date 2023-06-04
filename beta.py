import os
import numpy as np
import pandas as pd
import argparse

def calculate():
    parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting beta calculating')

    parser.add_argument('--root_path', type=str, default='./data/stocks_10/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='DS.csv', help='data file')
    parser.add_argument('--data_path2', type=str, default='DS.csv', help='data file')
    parser.add_argument('--target', type=str, default='Close', help='target feature in S or MS task')

    parser.add_argument('--seq_len', type=int, default=64, help='input sequence length of Informer encoder') # 120
    parser.add_argument('--label_len', type=int, default=40, help='start token length of Informer decoder') # 60 
    parser.add_argument('--pred_len', type=int, default=5, help='prediction sequence length') # 15
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')

    args = parser.parse_args()

    df_raw = pd.read_csv(os.path.join(args.root_path, args.data_path))
    df_raw2 = pd.read_csv(os.path.join(args.root_path, args.data_path))

    # cols = list(df_raw.columns); cols.remove(args.target)
    # df_raw = df_raw[cols+[args.target]]

    df_raw_len = len(df_raw) - args.pred_len
    num_train = int(df_raw_len * 0.7)
    num_test = int(df_raw_len * 0.2)
    num_vali = df_raw_len - num_train - num_test

    # border1s = [
    #     0,
    #     num_train - args.seq_len,
    #     df_raw_len - num_test - args.seq_len,
    #     df_raw_len - args.seq_len - args.batch_size + 1
    # ]
    # border2s = [
    #     num_train,
    #     num_train + num_vali,
    #     df_raw_len,
    #     df_raw_len + args.pred_len
    # ]

    df_data1 = df_raw[[args.target]].to_numpy().flatten()
    df_data2 = df_raw2[[args.target]].to_numpy().flatten()
    # train_data = df_data[border1s[0]:border2s[0]]
    train_data1 = df_data1[0:num_train]
    train_data2 = df_data2[0:num_train]
    correlation = np.corrcoef(train_data1, train_data2)[0][1]
    print("correlation: ", correlation)
    print("mean of data1: ", np.mean(train_data1))
    print("std of data1: ", np.std(train_data1))
    print("mean of data2: ", np.mean(train_data2))
    print("std of data2: ", np.std(train_data2))


if __name__ == '__main__':
    calculate()
import os
import pandas as pd
from bat.log_to_dataframe import LogToDataFrame


def bro_log_to_df(file_path):
    return LogToDataFrame(file_path)


def bro_logs_to_df(top_level_dir, log_kind):
    logs = make_log_list(top_level_dir, log_kind)
    df = pd.DataFrame()
    for log in logs:
        log_df = bro_log_to_df(log)
        # TODO: Fix hack to get to a pandas.DataFrame from a bat.log_to_dataframe.LogToDataFrame
        pandas_df = log_df.dropna(axis=1)
        # Note: pandas_df.shape == log_df.shape
        df = df.append(pandas_df)
    return df


def make_log_list(top_level_dir, log_kind):
    paths = []                                                           
    for rs, ds, fs in os.walk(top_level_dir):
        paths.extend([os.path.join(top_level_dir, f) for f in fs if log_kind in f])
    return paths


if __name__ == '__main__':
    bro_log_path = '../data/dns.06:00:00-07:00:00.log'
    bro_log_df = bro_log_to_df(bro_log_path)
    print(bro_log_df.shape)
    print(type(bro_log_df))
    print(bro_log_df.columns)

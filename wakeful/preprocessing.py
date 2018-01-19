"""
Get set of normal and attack Bro logs and create a balanced dataset.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from . import log_munger, metrics


def pipeline(dir_pairs, log_types):

    kv_pairs = []
    for norm_dir, mal_dir in dir_pairs:
        norm_base, mal_base = os.path.basename(norm_dir), os.path.basename(mal_dir)
        # TODO update functions that mutate dataframe to not have return statements

        for log_type in log_types:
            logs = log_munger.make_log_list(norm_dir, log_type)
            if not logs:
                continue
            norm_df = log_munger.bro_logs_to_df(norm_dir, log_type, logs=logs)
            logs = log_munger.make_log_list(mal_dir, log_type)
            if not logs:
                continue
            mal_df = log_munger.bro_logs_to_df(mal_dir, log_type, logs=logs)
            label_data(norm_df, mal_df)
            df = combine_dfs(norm_df, mal_df)

            if log_type.lower() == 'conn':
                df['pcr'] = metrics.calc_pcr(df)
            elif log_type.lower() == 'dns':
                df['query_entropy'] = df['query'].apply(metrics.calc_entropy)

            train, test = train_test_rebalance_split(df)
            kv_pairs.append((norm_base + '-' + mal_base + '-' + log_type, (train, test)))

    return dict(kv_pairs)


def label_data(norm_df, mal_df):
    norm_df['label'] = 0
    mal_df['label'] = 1


def combine_dfs(norm_df, mal_df):
    mal_col_set = set(mal_df.columns)
    norm_col_set = set(norm_df.columns)

    # remove inconsistent columns
    norm_columns_not_in_mal = {col_n for col_n in norm_col_set if col_n not in mal_col_set}
    norm_col_set = set(norm_df.columns)
    norm_df = norm_df.drop(norm_columns_not_in_mal, axis=1)

    # cover where either can have more columns
    mal_col_set = set(mal_df.columns)
    mal_columns_not_in_norm = {col_m for col_m in mal_col_set if col_m not in norm_col_set}
    mal_df = mal_df.drop(mal_columns_not_in_norm, axis=1)

    # append the data sets
    return norm_df.append(mal_df)


def train_test_rebalance_split(df):
    train, test = train_test_split(df, random_state=37, test_size=0.5)
    train_rebalanced = log_munger.rebalance(train, column_name='label')
    return train_rebalanced, test

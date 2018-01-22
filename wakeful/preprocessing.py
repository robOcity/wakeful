"""
Get set of normal and attack Bro logs and create a balanced dataset.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from . import log_munger, metrics

CONN_LOG_DROP_COLS = cols_to_drop = ['conn_state', 'history', 'id.orig_h',
              'id.resp_h', 'id.orig_p', 'id.resp_p', 'missed_bytes', 'proto', 'service',
              'tunnel_parents', 'uid', 'duration', 'uid']

DNS_LOG_DROP_COLS = ['TTLs', 'answers', 'AA', 'TC', 'RD', 'RA', 'Z', 'rejected',
                     'qclass_name', 'qtype', 'qclass', 'qtype_name', 'query', 'rcode',
                     'rcode_name', 'trans_id', 'id.orig_h', 'id.orig_p', 'id.resp_h',
                     'id.resp_p', 'uid']


def preprocess(dir_pairs, log_types):
    MIN_SAMPLES = 100
    # TODO update functions that mutate dataframe to not have return statements
    kv_pairs = []
    for mal_dir, norm_dir in dir_pairs:
        norm_base, mal_base = os.path.basename(norm_dir), os.path.basename(mal_dir)

        for log_type in log_types:
            # process logs representing normal traffic
            logs = log_munger.make_log_list(norm_dir, log_type)
            if not logs:
                continue
            norm_df = log_munger.bro_logs_to_df(norm_dir, log_type, logs=logs)

            # process logs representing attack trafficdf.
            logs = log_munger.make_log_list(mal_dir, log_type)
            if not logs:
                continue
            mal_df = log_munger.bro_logs_to_df(mal_dir, log_type, logs=logs)

            # add class labels to the data
            norm_df, mal_df = label_data(norm_df, mal_df)

            # combine the normal and attack dataframes into one
            df = combine_dfs(norm_df, mal_df)

            # add feature engineered columns and drop not used in modeling
            if log_type.lower() == 'conn':
                print('CONN', df.columns)
                df['pcr'] = metrics.calc_pcr(df)
                df['is_ipv4_host'] = df['id.orig_h'].apply(metrics.is_ipv4)
                df['is_ipv6_host'] = df['id.orig_h'].apply(metrics.is_ipv6)
                df['is_ipv4_resp'] = df['id.resp_h'].apply(metrics.is_ipv4)
                df['is_ipv6_resp'] = df['id.resp_h'].apply(metrics.is_ipv6)
                drop_columns(df, cols_to_drop=CONN_LOG_DROP_COLS)
            elif log_type.lower() == 'dns':
                print('DNS', df.columns)
                df['query_entropy'] = df['query'].apply(metrics.calc_entropy)
                df['query_length'] = df['query'].apply(metrics.calc_query_length)
                df['query_entropy'] = df['answers'].apply(metrics.calc_answer_length)
                df['is_ipv4_host'] = df['id.orig_h'].apply(metrics.is_ipv4)
                df['is_ipv6_host'] = df['id.orig_h'].apply(metrics.is_ipv6)
                df['is_ipv4_resp'] = df['id.resp_h'].apply(metrics.is_ipv4)
                df['is_ipv6_resp'] = df['id.resp_h'].apply(metrics.is_ipv6)
                drop_columns(df, cols_to_drop=DNS_LOG_DROP_COLS)

            # rebalance the training data
            if min(get_class_counts(df)) < MIN_SAMPLES:
                continue
            train, test = train_test_rebalance_split(df)
            kv_pairs.append((mal_base + '-' + norm_base + '-' + log_type, (train, test)))

    return dict(kv_pairs)


def label_data(norm_df, mal_df, label='label'):
    norm_df[label] = 0
    mal_df[label] = 1
    return norm_df, mal_df


def get_class_counts(df, label='label', pos_class=1, neg_class=0):
    pos_count = len(df[label][df[label] == pos_class])
    neg_count = len(df[label][df[label] == neg_class])
    return pos_count, neg_count


def drop_columns(df, cols_to_drop=None):
    df.drop(cols_to_drop, inplace=True, axis=1)


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

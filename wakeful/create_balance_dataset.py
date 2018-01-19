"""
Get set of normal and attack Bro logs and create a balanced dataset.
"""
from wakeful import log_munger, metrics


def build_df(dir_path, log_types):
    src_dir = os.path.basename(dir_path)
    kv_pairs = []
    for log_type in log_types:
        df = log_munger.bro_log_to_df(dir_path, log_type)
        df = add_features(df)
        df = clean_df(df)
        df = balance_df(df)
        kv_pairs.append((rc_dir + log_type, df))
    return dict(kv_pairs)


def add_features(df):
    df['pcr'] = metrics.calc_pcr(df)
    df['query_entropy'] = df['query'].apply(metrics.calc_entropy)
    return df


def clean_df(df):
    pass


def persist_df_dict(key, dir_path, df_dict):
    pass


def balance_df(df_dict):
    for name, df in df_dict.items():


if __name__ == '__main__':
    df_dict = {}
    log_types = ['dns', 'conn']
    df_dict.update(build_df('./data/home/2017-12-31', log_types))
    df_dict.update(build_df('./data/c2/dnscat2', log_types))
    df_dict.update(build_df('./data/c2/icmptunnel', log_types))
    df_dict.update(build_df('./data/c2/iodine-forwarded', log_types))
    df_dict.update(build_df('./data/c2/iodine-raw', log_types))
    df_dict.update(build_df('./data/c2/ptunnel', log_types))

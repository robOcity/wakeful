from wakeful import log_munger, pipelining


def fig_title(h5_key):
    tokens = [tok.capitalize() for tok in h5_key.split('_')]
    return f'Malware: {tokens[0]} -- Log Data Analyzed: {tokens[-2]}'


def fig_name(h5_key):
    tokens = [tok.capitalize() for tok in h5_key.split('_')]
    return f'{tokens[0]}_{tokens[-2]}.png'


if __name__ == '__main__':

    log_types = ['dns', 'conn']

    persisted_df_paths = dict([
        ('dnscat2_2017_12_31_conn_test', 'dnscat2_2017_12_31_conn_train'),
        ('dnscat2_2017_12_31_dns_test', 'dnscat2_2017_12_31_dns_train'),
        ('iodine_forwarded_2017_12_31_conn_test', 'iodine_forwarded_2017_12_31_conn_train'),
        ('iodine_forwarded_2017_12_31_dns_test', 'iodine_forwarded_2017_12_31_dns_train'),
        ('iodine_raw_2017_12_31_conn_test', 'iodine_raw_2017_12_31_conn_train'),
        ('iodine_raw_2017_12_31_dns_test', 'iodine_raw_2017_12_31_dns_train'),
    ])

    data_dir = 'data/'
    fig_dir = 'plot/'
    for test_key, train_key in persisted_df_paths.items():
        train_df = log_munger.hdf5_to_df(train_key, data_dir)
        test_df = log_munger.hdf5_to_df(test_key, data_dir)
        df_dict = pipelining.feature_selection_pipeline(train_df=train_df,
                                                        test_df=test_df,
                                                        fig_dir=fig_dir,
                                                        fig_file=fig_name(test_key),
                                                        fig_title=fig_title(test_key))

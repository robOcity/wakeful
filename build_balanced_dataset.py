from wakeful import preprocessing, log_munger


if __name__ == '__main__':

    log_types = ['dns', 'conn']

    dir_pairs = [('./data/c2/dnscat2', './data/home/2017-12-31'),
                 ('./data/c2/icmptunnel', './data/home/2017-12-31'),
                 ('./data/c2/iodine-forwarded', './data/home/2017-12-31'),
                 ('./data/c2/iodine-forwarded', './data/home/2017-12-31'),
                 ('./data/c2/iodine-raw', './data/home/2017-12-31'),
                 ('./data/c2/ptunnel', './data/home/2017-12-31')]

    df_dict = preprocessing.preprocess(dir_pairs, log_types)

    for name, df in df_dict.items():
        dir_path = './data'
        train, test = df
        log_munger.df_to_hdf5(train, name + '-train', dir_path)
        log_munger.df_to_hdf5(test, name + '-test', dir_path)
        print()
        print(name, 80*'-')
        print(train.columns)

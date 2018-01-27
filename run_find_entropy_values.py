import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from wakeful import preprocessing, pipelining, scoring, log_munger


def main():
    data_dir = './data/'

    keys = [
        ('iodine_forwarded_2017_12_31_dns_test', 'iodine_forwarded_2017_12_31_dns_train'),
        ('iodine_raw_2017_12_31_dns_test', 'iodine_raw_2017_12_31_dns_train'),
        ('dnscat2_2017_12_31_dns_test', 'dnscat2_2017_12_31_dns_train')]

    results = []
    for test_key, train_key in keys:
        # read in the persisted dataframes
        train_df = log_munger.hdf5_to_df(train_key, data_dir)
        test_df = log_munger.hdf5_to_df(test_key, data_dir)

        df

    # create dataframe
    print(results)
    df = pd.DataFrame.from_records(results)
    print(df.head())




if __name__ == '__main__':
    main()

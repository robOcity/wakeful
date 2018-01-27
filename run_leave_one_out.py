import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from wakeful import preprocessing, pipelining, scoring, log_munger


def main():
    data_dir = './data/'

    keys = [
        ('iodine_forwarded_2017_12_31_conn_test', 'iodine_forwarded_2017_12_31_conn_train'),
        ('iodine_raw_2017_12_31_conn_test', 'iodine_raw_2017_12_31_conn_train'),
        ('dnscat2_2017_12_31_conn_test', 'dnscat2_2017_12_31_conn_train'),
        ('iodine_forwarded_2017_12_31_dns_test', 'iodine_forwarded_2017_12_31_dns_train'),
        ('iodine_raw_2017_12_31_dns_test', 'iodine_raw_2017_12_31_dns_train'),
        ('dnscat2_2017_12_31_dns_test', 'dnscat2_2017_12_31_dns_train')]

    results = []
    for test_key, train_key in keys:
        # read in the persisted dataframes
        train_df = log_munger.hdf5_to_df(train_key, data_dir)
        test_df = log_munger.hdf5_to_df(test_key, data_dir)

        # list of dicts with scoring of leave-one-out modeling results
        print(type(test_df))
        print(test_df.head())

        result = leave_one_out(train_df, test_df, test_key)

        # grow the list
        results.extend(result)

    # create dataframe
    print(results)
    df = pd.DataFrame.from_records(results)
    print(df.head())

    # make plot using seaborn box plot
    # Set up the matplotlib figure
    # f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    # y2 = y1 - 5
    # sns.barplot(x, y2, palette="RdBu_r", ax=ax2)
    # ax2.set_ylabel("Diverging")


def leave_one_out(train_df, test_df, key):
    results = []
    print(test_df.columns)
    for feature_removed in test_df.columns:

        # remove one feature and see how the models do
        train_df = train_df.drop(feature_removed, axis=1)
        test_df = test_df.drop(feature_removed, axis=1)

        # drop rows with empty values
        train_df = train_df.dropna(axis=0, how='any')
        test_df = test_df.dropna(axis=0, how='any')

        # split the datasets
        X_train, y_train = preprocessing.split_X_y(train_df)
        X_test, y_test = preprocessing.split_X_y(test_df)

        # classifiers to evaluate
        models = {
            'gradient boosting':   GradientBoostingClassifier(n_estimators=200, min_samples_leaf=10),
            'logistic regression': LogisticRegression(C=1e5),
        }

        for estimator_name, estimator in models.items():

            pipe, trained_estimator = pipelining.modeling_pipeline(estimator=estimator,
                                                            X_train=X_train,
                                                            y_train=y_train)

            result = scoring.print_scores(estimator_name=estimator_name,
                                 data_name=key,
                                 estimator=trained_estimator,
                                 X_test=X_test.values,
                                 y_test=y_test.values)

            result.update({'feature_removed': feature_removed, 'estimator': estimator_name})
            results.append(result)

    return result


if __name__ == '__main__':
    main()

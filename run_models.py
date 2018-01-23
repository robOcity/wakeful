import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from wakeful import preprocessing, pipelining, scoring, log_munger


if __name__ == '__main__':
    data_dir = './data/'

    keys = [
        ('iodine-forwarded-2017-12-31-conn-test', 'iodine-forwarded-2017-12-31-conn-train'),
        ('iodine-raw-2017-12-31-conn-test', 'iodine-raw-2017-12-31-conn-train'),
        ('dnscat2-2017-12-31-conn-test', 'dnscat2-2017-12-31-conn-train'),
        ('iodine-forwarded-2017-12-31-dns-test', 'iodine-forwarded-2017-12-31-dns-train'),
        ('iodine-raw-2017-12-31-dns-test', 'iodine-raw-2017-12-31-dns-train'),
        ('dnscat2-2017-12-31-dns-test', 'dnscat2-2017-12-31-dns-train'),]

    for test_key, train_key in keys:
        # read in the persisted dataframe
        train_df = log_munger.hdf5_to_df(train_key, data_dir)
        test_df = log_munger.hdf5_to_df(test_key, data_dir)

        # TODO move to preprocessing
        train_df = train_df.dropna(axis=0, how='any')
        test_df = test_df.dropna(axis=0, how='any')

        # split the datasets
        X_train, y_train = preprocessing.split_X_y(train_df)
        X_test, y_test = preprocessing.split_X_y(test_df)

        # classifiers to evaluate
        models = {
            'gradient boosting': GradientBoostingClassifier(n_estimators=200, min_samples_leaf=10),
            'random forest': RandomForestClassifier(n_estimators=100, min_samples_leaf=5),
            'logistic regression': LogisticRegression(C=1e5),
        }

        for estimator_name, estimator in models.items():

            pipe, trained_estimator = pipelining.modeling_pipeline(estimator=estimator,
                                                            X_train=X_train,
                                                            y_train=y_train)

            scoring.print_scores(estimator_name=estimator_name,
                                 data_name=train_key[:train_key.rfind('-')],
                                 estimator=trained_estimator,
                                 X_test=X_test.values,
                                 y_test=y_test.values)

"""
All the scikit learn pipeline functionality and classes used in the
project are kept here.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from . import log_munger


def modeling_pipeline(key, data_dir):

    # read in the persisted dataframe
    df = log_munger.hdf5_to_df(key, data_dir)

    # separate the labels from the data
    y = df.pop('label')
    X = df

    # create the transformers and estimators
    encoder = DummyEncoder()
    fillna = Imputer(strategy='median')
    scaler = StandardScaler()
    knn = KNeighborsClassifier(n_neighbors=5)
    sfs = SFS(knn,
              k_features=5,
              forward=True,
              floating=False,
              verbose=2,
              scoring='recall',
              cv=3)
    # TODO - Figure out how to use a Feature Union to run these models in parallel using best features
    boosting = GradientBoostingClassifier(n_estimators=100, min_samples_leaf=10)
    # randomforest = RandomForestClassifier(n_estimators=100, min_samples_leaf=5)
    # logistic = LogisticRegression()

    pipe = Pipeline(steps=[
        ('encoder', encoder),
        ('fillna', fillna),
        ('scaler', scaler),
        ('sfs', sfs),
        ('boosting', boosting),
    ])

    pipe.fit(X, y)

    print(80 * '-')
    print(pd.DataFrame.from_dict(sfs.get_metric_dict()).T)
    print()

    print(80 * '-')
    print(pd.DataFrame.from_dict(encoder.matrix_lookup_, orient='index').T)
    print()

    print('Selected features:', sfs.k_feature_idx_)

    fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')
    plt.title('Sequential Forward Selection (w/ standard error)')
    plt.grid()
    plt.show()

    # print('best combination (ACC: %.3f): %s\n' % (sfs.k_score_, sfs.k_feature_idx_))
    # print('all subsets:\n', sfs.subsets_)
    # plot_sfs(sfs.get_metric_dict(), kind='f1')
    # plt.show()


class DummyEncoder(TransformerMixin, BaseEstimator):
    """
    Converts pandas dataframes to numpy matrices and back again.
    Based on code from Tom Augsperger's github repository
    (https://github.com/TomAugspurger/mtg) and his talk at
    PyData Chicago 2016 (https://youtu.be/KLPtEBokqQ0).  Added a map
    of columns in the dataframe to those in the matrix.  This information
    is held in the DummyEncoder.matrix_lookup -- a dictionary
    keyed by the matrix column index with values showing the column
    from the dataframe.  For dummy encoded dataframe columns it shows
    column.encoded_value.
    """
    def fit(self, X, y=None):
        # record info here, use in transform, inv_transform.
        self.columns_ = X.columns
        self.cat_cols_ = X.select_dtypes(include=['category']).columns
        self.non_cat_cols_ = X.columns.drop(self.cat_cols_)

        self.cat_map_ = {col: X[col].cat for col in self.cat_cols_}
        left = len(self.non_cat_cols_) # 2
        self.cat_blocks_ = {}

        for col in self.cat_cols_:
            right = left + len(X[col].cat.categories)
            self.cat_blocks_[col] = slice(left, right)
            left = right

        # provide clear relationship between columns in encoded matrix
        # columns and the dataframe
        cat_matrix_cols = [f'{k}.{v}'
                           for k, v in self.cat_map_.items()
                           for v in v.categories.get_values()]
        all_matrix_cols = list(self.non_cat_cols_.get_values()) + cat_matrix_cols
        self.matrix_lookup_ = {i:v for i, v in enumerate(all_matrix_cols)}

        return self

    def transform(self, X, y=None):
        return np.asarray(pd.get_dummies(X))

    def inverse_transform(self, trn, y=None):
        # Numpy to Pandas DataFrame
        # original column names <=> positions
        numeric = pd.DataFrame(trn[:, :len(self.non_cat_cols_)],
                               columns=self.non_cat_cols_)
        series = []
        for col, slice_ in self.cat_blocks_.items():
            codes = trn[:, slice_].argmax(1)
            cat = pd.Categorical.from_codes(codes,
                                            self.cat_map_[col].categories,
                                            ordered=self.cat_map_[col].ordered)
            series.append(pd.Series(cat, name=col))
        return pd.concat([numeric] + series, axis=1)[self.columns_]

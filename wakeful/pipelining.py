"""
All the scikit learn pipeline functionality and classes used in the
project are kept here.
"""

import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
# from mlxtend.feature_selection import SequentialFeatureSelector as SFS
# from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from . import preprocessing
import matplotlib
import matplotlib.pyplot as plt


# def plot_bar_chart(plotting_df):
#     f, ax = plt.subplots(1, 1, figsize=(7, 5))
#     plt.title('Feature Selection')
#     ax.set_xlim([0, 1])
#     # sns.set_style("whitegrid", { 'axes.edgecolor': '.8', 'text.color': '.15', 'xtick.color': '.15',})
#     #sns.barplot(x='importance', 
#                 y='feature', 
#                 data=plotting_df, 
#                 palette=sns.color_palette('BuGn_r'))
#     # ax.set_xlabel('Importance', fontsize=12)
#     # ax.set_ylabel('Feature', fontsize=12)
#     #sns.despine(offset=10)
#     with plt.style.context('seaborn-poster'):
#         plt.tight_layout(pad=0.8)
#         plt.show()


# def plot_feature_importance(names, values):
#     f, ax = plt.subplots(1, 1, figsize=(7, 5))
#     pos = np.arange(len(names)) + 0.5
#     plt.barh(pos, values, align='center')
#     plt.title("Feature Importance")
#     plt.xlabel("Model Accuracy")
#     plt.ylabel("Features")
#     plt.yticks(pos, names)
#     plt.grid(True)
#     plt.tight_layout(pad=0.9)
#     plt.show()


# def feature_selection_pipeline(train_df=None, test_df=None, fig_dir=None, fig_file=None, fig_title=None):

#     # get features and labels
#     X_train, y_train = preprocessing.split_X_y(train_df)

#     # create the transformers and estimators
#     #encoder = DummyEncoder()
#     fillna = Imputer(strategy='median')
#     scaler = StandardScaler()
#     extree = ExtraTreesClassifier()

#     # knn = KNeighborsClassifier(n_neighbors=5)
#     # sfs = SFS(knn,
#     #           k_features=3,
#     #           forward=True,
#     #           floating=False,
#     #           verbose=2,
#     #           scoring='recall',
#     #           cv=3)

#     pipe = Pipeline(steps=[
#         #('encoder', encoder),
#         ('fillna', fillna),
#         ('scaler', scaler),
#         ('extree', extree)
#         #('sfs', sfs),
#     ])

#     # fit the pipe start to finish
#     pipe.fit(X_train, y_train)

#     # plot feature 
#     column_names = X_train.columns.values
#     labels = ['feature', 'importance']
#     data = [(name, value) for name, value in zip(column_names, extree.feature_importances_)]
#     plotting_df = pd.DataFrame.from_records(data, columns=labels)
#     print(plotting_df)
#     print(len(column_names), len(extree.feature_importances_))
#     plot_feature_importance(column_names, extree.feature_importances_)
    # plot_bar_chart(plotting_df.sort_values(by='importance', ascending=False))

    # produce output
    # print(f'{fig_title :_<80}')
    # print(pd.DataFrame.from_dict(sfs.get_metric_dict()).T)
    # print()

    # print(100 * '-')
    # print(pd.DataFrame.from_dict(encoder.matrix_lookup_, orient='index').T)
    # print()

    # print(100 * '-')
    # print('selected features:', sfs.k_feature_idx_)
    # print('\nbest combination: (ACC: %.3f): %s\n' % (sfs.k_score_, sfs.k_feature_idx_))
    # print('all subsets:\n', sfs.subsets_)

    # plt.title(fig_title)
    # plt.grid()
    # plot_sfs(sfs.get_metric_dict(), kind='std_err')
    # plt.savefig(os.path.join(fig_dir, fig_file), dpi=300)
    # plt.close()



def modeling_pipeline(estimator=None, X_train=None, y_train=None):
    # objects to prepare the data
    encoder = DummyEncoder()
    fillna = Imputer(strategy='median')
    scaler = StandardScaler()

    pipe = Pipeline(steps=[
        ('encoder', encoder),
        ('fillna', fillna),
        ('scaler', scaler),
        ('estimator', estimator),
    ])
    pipe.fit(X_train, y_train)
    return pipe, estimator


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
        self.cat_columns_ = X.select_dtypes(include=['category']).columns
        self.non_cat_columns_ = X.columns.drop(self.cat_columns_)

        self.cat_map_ = {col: X[col].cat for col in self.cat_columns_}
        left = len(self.non_cat_columns_)  # 2
        self.cat_blocks_ = {}

        for col in self.cat_columns_:
            right = left + len(X[col].cat.categories)
            self.cat_blocks_[col] = slice(left, right)
            left = right

        # provide clear relationship between columns in encoded matrix
        # columns and the dataframe
        cat_matrix_cols = [f'{k}.{v}'
                           for k, v in self.cat_map_.items()
                           for v in v.categories.get_values()]
        all_matrix_cols = list(self.non_cat_columns_.get_values()) + cat_matrix_cols
        self.matrix_lookup_ = {i: v for i, v in enumerate(all_matrix_cols)}

        return self

    def transform(self, X, y=None):
        return np.asarray(pd.get_dummies(X))

    def inverse_transform(self, trn, y=None):
        # Numpy to Pandas DataFrame
        # original column names <=> positions
        numeric = pd.DataFrame(trn[:, :len(self.non_cat_columns_)],
                               columns=self.non_cat_columns_)
        series = []
        for col, slice_ in self.cat_blocks_.items():
            codes = trn[:, slice_].argmax(1)
            cat = pd.Categorical.from_codes(codes,
                                            self.cat_map_[col].categories,
                                            ordered=self.cat_map_[col].ordered)
            series.append(pd.Series(cat, name=col))
        return pd.concat([numeric] + series, axis=1)[self.columns_]

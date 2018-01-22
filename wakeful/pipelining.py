"""
All the scikit learn pipeline functionality and classes used in the
project are kept here.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


def modeling_pipeline():
    pipe = make_pipeline(

    )


class DummyEncoder(TransformerMixin, BaseEstimator):
    """
    Converts pandas dataframes to numpy matrices and back again.
    Based on code from Tom Augsperger's github repository
    (https://github.com/TomAugspurger/mtg) and his talk at
    PyData Chicago 2016 (https://youtu.be/KLPtEBokqQ0).  Added a map
    of columns in the dataframeto those in the matrix.  This information
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
                           for k, v in de.cat_map_.items()
                           for v in v.categories.get_values()]
        all_matrix_cols = list(de.non_cat_cols_.get_values()) + cat_matrix_cols
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
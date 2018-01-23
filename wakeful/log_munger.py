import os
import glob
import pandas as pd
from bat.log_to_dataframe import LogToDataFrame
from sklearn.model_selection import train_test_split
from collections import defaultdict



def bro_log_to_df(file_path):
    """
    Load a Bro log in to a pandas DataFrame object.
    :param file_path: Log file
    :return: pandas DataFrame object
    """
    return LogToDataFrame(file_path)


def bro_logs_to_df(top_level_dir, log_type, logs=None):
    """
    Recursively find log files and load them into a DataFrame object.  Only log files
    that contain the given log name will be loaded.
    :param top_level_dir: Root directory for the recursive search
    :param log_type: Name of the Bro log file to load (e.g., dns)
    :return:
    """
    if not logs:
        logs = make_log_list(top_level_dir, log_type)
    df = pd.DataFrame()
    for log in logs:
        log_df = bro_log_to_df(log)
        # TODO: Get a pandas reference, not a bat reference, bat objects don't functions as dataframes
        pandas_df = log_df.dropna(axis=1) # hack to get a pandas dataframe with ALL columns
        # Note: pandas_df.shape == log_df.shape
        df = df.append(pandas_df)
    return df


def make_log_list(log_root_dir, log_type):
    """
    Builds a list of Bro log files by recursively searching all subdirectories under
    the root directory.
    :param top_level_dir: Root directory for the recursive search
    :param log_type: Name of the Bro log file to load (e.g., dns)
    :return: List of paths to log file that were found
    """
    cwd = os.getcwd()
    path = os.path.join(f'{log_root_dir}', f'**/*{log_type}*.log')
    results = glob.glob(path, recursive=True)
    return results


def df_to_hdf5(df, key, dir_path):
    """
    Save the DataFrame object as an HDF5 file. The file is stored
    in the directory specified and uses the key for the filename
    and 'h5' as the extension.
    :param df: DataFrame to save as a file
    :param key: ID for storage and retrieval
    :param dir_path: Directory to store the HDF5 data file
    """
    file_path = os.path.join(dir_path, key + '.h5')
    df.to_hdf(file_path, key, complevel=9, complib='zlib')
    return file_path


def hdf5_to_df(key, dir_path):
    """
    Retrieve the persisted DataFrame object from the HDF5 file.
    :param dir_path: Directory to store the HDF5 data file
    :return: Retrieved pandas DataFrame object
    """
    # TODO DRY --> make into function
    file_path = os.path.join(dir_path, key + '.h5')
    return pd.read_hdf(file_path, key)


def split_train_test(df, test_size=0.5):
    """
    Split the dataframe into a training and test in the desired proportion.
    :param df: pandas dataframe to split_train_test
    :param test_size: proportion of the sample to allocate to test
    :return: train, test dataframes
    """
    # split the data
    return train_test_split(df, random_state=RAND_SEED_VAL, test_size=test_size)


def split_X_y(df, column_name='label'):
    """
    Split the data into a feature values (X) and targets or labels (y).
    :param df: pandas dataframe to split
    :param target: Name of the column holding the labels
    :return: tuple containing X, y values
    """
    if column_name not in df.columns:
        return df, None
    y = df.pop([column_name])
    return df, y


def find_columns_by_type(df, type):
    """
    Finds the columns in a pandas dataframe of a particular type.
    :param df:  pandas dataframe to searching
    :param type: type of columns to find ('int64', 'float42', 'object', 'timedelta64[ns]', ...)
    :return: list of columns names of the type requested, or an empty list if none are found.
    """
    df_types = defaultdict(list)
    grp = df.columns.to_series().groupby(df.dtypes).groups
    df_types.update({k.name: v.values for k, v in grp.items()})
    return df_types.get(type)


def build_labeled_set(class_0_dir, class_1_dir):
    pass

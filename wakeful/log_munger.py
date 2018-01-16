import os
import glob
import pandas as pd
from bat.log_to_dataframe import LogToDataFrame
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

RAND_SEED_VAL = 42

def bro_log_to_df(file_path):
    """
    Load a Bro log in to a pandas DataFrame object.
    :param file_path: Log file
    :return: pandas DataFrame object
    """
    return LogToDataFrame(file_path)


def bro_logs_to_df(top_level_dir, log_type):
    """
    Recursively find log files and load them into a DataFrame object.  Only log files
    that contain the given log name will be loaded.
    :param top_level_dir: Root directory for the recursive search
    :param log_type: Name of the Bro log file to load (e.g., dns)
    :return:
    """
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


def generate_pcr_feature(df, column_name='pcr'):
    """
    Calculate the producer-consumer-ratio for the dataframe.  The dataframe needs
    to have the orig_bytes and resp_bytes fields from the th conn log.  If so,
    a new 'pcr' column will be added, otherwise the dataframe will remain unchanged.
    :param df: Dataframe to mutate
    :param column_name: Column name holding the PCR values (default is 'pcr')
    """
    if 'orig_bytes' in df.columns and 'resp_bytes' in df.columns:
        numerator = (df.orig_bytes - df.resp_bytes)
        denominator = (df.orig_bytes + df.resp_bytes)
        df[column_name] = numerator / denominator


def df_to_hdf5(df, key, dir_path):
    """
    Save the DataFrame object as an HDF5 file.
    :param df: DataFrame to save as a file
    :param key: ID for storage and retrieval
    :param dir_path: Path to directory of where to save the file
    """
    file_path = os.path.join(dir_path, key + '.h5')
    df.to_hdf(file_path, key, complevel=9, complib='zlib')


def hdf5_to_df(dir_path, key):
    """
    Retrieve the persisted DataFrame object from the HDF5 file.
    :param path: Path to directory of where to save the file
    :return: Retrieved DataFrame object
    """
    file_path = os.path.join(dir_path, key + '.h5')
    return pd.read_hdf(file_path, key)


def rebalance(df, column_name='label'):
    """
    Adds minority class values by resampling with replacement.
    Apply this function to TRAINING data only, not testing!
    :param df: pandas dataframe to balance
    :param target: Name of the column holding the labels
    :return: rebalanced data frame with rows added to the minority class
    """
    # split the data
    train, test = train_test_split(df, random_state=RAND_SEED_VAL, test_size=0.5)

    # balance classes by upsampling with replacement
    train_majority = train[train[column_name]==0]
    train_minority = train[train[column_name]==1]
    train_deficit = len(train_majority) - len(train_minority)
    train_minority_rebalanced = resample(train_minority,
                                    replace=True,
                                    random_state=RAND_SEED_VAL,
                                    n_samples=len(train_majority))

    # create the balanced data set
    return train_majority.append(train_minority_rebalanced)


def split_X_y(df, column_name='label'):
    """
    Split the data into a feature values (X) and targets or labels (y).
    :param df: pandas dataframe to split
    :param target: Name of the column holding the labels
    :return: tuple containing X, y values
    """
    y = df.pop([column_name])
    return df, y


def find_column_by_type(df, type):
    """Finds the columns in a pandas dataframe of a particular type.
    :param df:  pandas dataframe to searching
    :param type: type of columns to find ('int64', 'float42', 'object', 'timedelta64', ...)
    :return: list of columns names of the type requested, or an empty list if none are found.
    """
    df_types = defaultdict(list)
    grp = df.columns.to_series().groupby(df.dtypes).groups
    df_types.update({k.name: v.values for k, v in grp.items()})
    return list(df[type])


def calc_prc(df, src_bytes_col='orig_bytes', dest_bytes_col='resp_bytes'):
    """
    Calculate the producer-consumer ratio for network connections.  The PCR is ratio
    is -1.0 for consumers and 1.0 for producers.  PCR is independent from the speed
    of the network.
    :param df: pandas dataframe
    :param src_bytes_col: name of column with the number of bytes sent by the source
    :param dest_bytes_col: column name with the number of bytes sent by the destination
    :return: pandas series containing the pcr values
    """
    numerator = norm_df.orig_bytes - norm_df.resp_bytes
    denominator = norm_df.orig_bytes + norm_df.resp_bytes
    result = numerator / denominator if denomerator != 0 and denominator != 0 else np.nan
    return result

import os
import glob
import pandas as pd
from bat.log_to_dataframe import LogToDataFrame


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
        # TODO: Improve getting pandas.DataFrame from bat.log_to_dataframe.LogToDataFrame
        pandas_df = log_df.dropna(axis=1)
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
    path = os.path.join(f'{log_root_dir}', f'**/*{log_type}*.log')
    results = glob.glob(path, recursive=True)
    return results


def df_to_hdf5(df, path, append=False):
    """
    Save the DataFrame object as an HDF5 file.
    :param df: DataFrame to save as a file
    :param path: Path to directory of where to save the file
    :param append: 'True' appends to the file, 'False' overwrites it
    """
    df.to_hdf(path, 'table', append=append)


def hdf5_to_df(path):
    """
    Retrieve the persisted DataFrame object from the HDF5 file.
    :param path: Path to directory of where to save the file
    :return: Retrieved DataFrame object
    """
    return pd.read_hdf(path, 'table')

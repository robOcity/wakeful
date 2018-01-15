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


if __name__ == '__main__':
    cwd = os.getcwd()
    print(cwd)

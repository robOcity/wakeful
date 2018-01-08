import os
import pandas as pd
import numpy as np
import pytest
from wakeful import log_munger


@pytest.fixture()
def data_dir():
    return '../data/home/2017-12-31/'


def test_bro_log_to_df(data_dir):
    file_path = os.path.join(data_dir, 'dns.06:00:00-07:00:00.log')
    bro_df = log_munger.bro_log_to_df(file_path)
    assert(isinstance(bro_df, pd.core.frame.DataFrame))
    assert((1085, 23) == bro_df.shape)


def test_make_log_list_one_dir(data_dir):
    log_type = 'dns'
    log_list = log_munger.make_log_list(os.path.join(data_dir), log_type)
    assert(len(log_list) == 23)
    assert(all((log_type in f for f in log_list)))


def test_bro_logs_to_df(data_dir):
    """
    The dns.log file has 23 columns
    Find number of lines in the log files:
    32020 <-- find data/home/2017-12-31 -type f -name "*dns*log" | xargs wc -l
    Bro logs have 9 lines that are not data per log (8 comments at the the top and 1 at the bottom)
    31813 <-- 32020 - 23 * 9
    So, I believe this result is correct
    """
    log_type = 'dns'
    bro_df = log_munger.bro_logs_to_df(os.path.join(data_dir), log_type)
    bro_df = log_munger.bro_logs_to_df(os.path.join(data_dir), log_type)
    assert((31813, 23) == bro_df.shape)


@pytest.fixture()
def expected_df():
    ROWS = 10
    return pd.DataFrame(np.random.randn(ROWS, 1),
                       columns=['A'],
                       index=pd.date_range('20170101', periods=ROWS, freq='T'))


@pytest.fixture(scope='session')
def test_dir(tmpdir_factory):
    test_dir = tmpdir_factory.mktemp('temp')
    return test_dir


@pytest.fixture(scope='session')
def persist_path(test_dir):
    test_filepath = test_dir.join('persistent_log_store.h5')
    return test_filepath


def test_df_to_hdf5(persist_path, expected_df):
    log_munger.df_to_hdf5(expected_df, persist_path)
    assert(os.path.isfile(persist_path))


def test_hdf5_to_df(persist_path, expected_df):
    log_munger.df_to_hdf5(expected_df, persist_path)
    actual_df = log_munger.hdf5_to_df(persist_path)
    assert(expected_df.shape == actual_df.shape)
    assert(expected_df.equals(actual_df))


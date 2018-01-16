import os
import pandas as pd
import numpy as np
import pytest
from wakeful import log_munger, metrics


@pytest.fixture()
def data_dir():
    return 'data/home/2017-12-31/'


@pytest.fixture()
def c2_data_dir():
    return 'data/c2'


def test_bro_log_to_df(data_dir):
    file_path = os.path.join(data_dir, 'dns.06:00:00-07:00:00.log')
    bro_df = log_munger.bro_log_to_df(file_path)
    assert(isinstance(bro_df, pd.core.frame.DataFrame))
    assert((1085, 23) == bro_df.shape)


def test_make_log_list(data_dir):
    log_type = 'conn'
    log_list = log_munger.make_log_list(os.path.join(data_dir), log_type)
    assert(len(log_list) == 23)
    assert(all((log_type in f for f in log_list)))


def test_make_log_list_recursive(c2_data_dir):
    log_type = 'conn'
    log_list = log_munger.make_log_list(c2_data_dir, log_type)
    assert(len(log_list) == 5)
    assert(all((log_type in f for f in log_list)))


def test_bro_logs_to_df(data_dir):
    """
    23 dns.log files <-- find data/home/2017-12-31 -type f -name "*dns*log" | wc -l
    32020 lines in these files <-- find data/home/2017-12-31 -type f -name "*dns*log" | xargs wc -l
    Bro logs have 9 comment lines per log (8 top + 1 bottom)
    Number of rows --expect --> 31813 <-- 32020 - 23 files * 9 comments / file
    Number of columns --expect --> 23 since that is the number of columns in the dns log
    """
    log_type = 'dns'
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


def test_df_to_hdf5(test_dir, expected_df):
    key = 'key_value_and_filename'
    filepath = log_munger.df_to_hdf5(expected_df, key, test_dir)
    assert(os.path.isfile(filepath))


def test_hdf5_to_df(test_dir, expected_df):
    key = 'key_value_and_filename'
    log_munger.df_to_hdf5(expected_df, key, test_dir)
    actual_df = log_munger.hdf5_to_df(key, test_dir)
    assert(expected_df.shape == actual_df.shape)
    assert(expected_df.equals(actual_df))


def test_calc_entropy_low():
    uniform = 'ccccccccccccc'
    assert(abs(metrics.calc_entropy(uniform)) == 0.0)


def test_calc_entropy_high():
    rubbish = 'bpaopw5h;lna v08pqo5iup6b2pw96fy09 yr4tp   i5h'
    assert(abs(metrics.calc_entropy(rubbish)) > 1.0)

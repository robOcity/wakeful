import os
import pandas as pd
import numpy as np
import pytest
from wakeful import log_munger


def test_bro_log_to_df():
    file_path = '../data/home/2017-12-31/dns.06:00:00-07:00:00.log'
    bro_df = log_munger.bro_log_to_df(file_path)
    assert(isinstance(bro_df, pandas.core.frame.DataFrame))
    assert((1085, 23) == bro_df.shape)


def test_make_log_list_one_dir():
    top_level_dir = '../data/home/2017-12-31/'
    log_kind = 'dns'
    log_list = log_munger.make_log_list(top_level_dir, log_kind)
    assert(len(log_list) == 23)
    assert(all((log_kind in f for f in log_list)))


def test_bro_logs_to_df():
    """
    The dns.log file has 23 columns
    Find number of lines in the log files:
    32020 <-- find data/home/2017-12-31 -type f -name "*dns*log" | xargs wc -l
    Bro logs have 9 lines that are not data per log (8 comments at the the top and 1 at the bottom)
    31813 <-- 32020 - 23 * 9
    So, I believe this result is correct
    """
    top_level_dir = '../data/home/2017-12-31/'
    log_kind = 'dns'
    bro_df = log_munger.bro_logs_to_df(top_level_dir, log_kind)
    assert((31813, 23) == bro_df.shape)


@pytest.fixture
def get_df():
    dft = pd.DataFrame(np.random.randn(100000, 1),
                       columns=['A'],
                       index=pd.date_range('20170101', periods=100000, freq='T'))
    return dft.to_json(orient='table')


@pytest.fixture
def test_dir(tmpdir_factory):
    test_dir = tmpdir_factory.mktemp('temp')
    return test_dir


@pytest.fixture
def json_path(test_dir):
    test_filepath = test_dir.join('test_df.json')
    return test_filepath


@pytest.fixture
def test_df_to_json(json_path, create_df):
    log_munger.df_to_json(get_df, json_path)
    assert(os.path.isfile(json_path))



# @pytest.fixture
# def test_df_to_json(tmpdir_factory):
#     persisted_json_path = test_dir.join('persisted')
#     log_munger.df_to_json(df, persisted_json_path)
#     assert(os.path.isfile(persisted_json_path))

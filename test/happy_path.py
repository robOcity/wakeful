import pytest
import pandas

from src import log_munger


def test_bro_to_df_dns():
    file_path = '../data/dns.06:00:00-07:00:00.log'
    bro_df = log_munger.bro_to_df(file_path)
    assert(isinstance(bro_df, pandas.core.frame.DataFrame))
    assert(bro_df.shape == (1196, 23))

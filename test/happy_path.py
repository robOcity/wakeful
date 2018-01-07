import pandas

from src import log_munger


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
    Bro logs 9 lines that are not dataa per log (8 comments lines at the the top and 1 at the bottom)
    31813 <-- 32020 - 23 * 9
    So, I believe this result is correct
    :return:
    """
    top_level_dir = '../data/home/2017-12-31/'
    log_kind = 'dns'
    bro_df = log_munger.bro_logs_to_df(top_level_dir, log_kind)
    assert((31813, 23) == bro_df.shape)


import os
import re
import datetime
import pytest
import pandas as pd
import numpy as np
import whois  # pip lists it as python-whois
from wakeful import log_munger, metrics


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


def test_calc_pcr(pseudo_conn_log_df):
    df = metrics.calc_pcr(pseudo_conn_log_df)
    assert(df.loc['a'] == pytest.approx(-1/3))


def test_calc_entropy_high():
    rubbish = 'bpaopw5h;lna v08pqo5iup6b2pw96fy09 yr4tp   i5h'
    assert(abs(metrics.calc_entropy(rubbish)) > 1.0)


def test_calc_url_rep(json_good_rep):
    assert(metrics.calc_url_reputation(json_good_rep) == 0.0)


def test_is_new_url():
    url = 'galvanize.com'
    w = whois.whois(url)
    created = w.creation_date
    one_day = datetime.timedelta(days=1)
    still_to_new = created + 99 * one_day
    just_now_ok = created + 101 * one_day
    assert(metrics.is_new_url(url, set_current_date=created.isoformat()) == True)


def test_is_ipv4():
    good = '192.168.7.10'
    bad = '256.100.200.200'
    assert(metrics.is_ipv4(good) == True)
    assert(metrics.is_ipv4(bad) == False)


def test_is_ipv6():
    good = '2001:db8::1'
    bad = 'fe80::1ff:fe23:4567::890a'
    assert(metrics.is_ipv6(good) == True)
    assert(metrics.is_ipv6(bad) == False)


# Fixtures -----------------------------------------------------------------

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


@pytest.fixture()
def pseudo_conn_log_df():
    return pd.DataFrame({'orig_bytes' : 5., 'resp_bytes' : 10.}, index=['a', 'b'])


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


@pytest.fixture()
def data_dir():
    return 'data/home/2017-12-31/'


@pytest.fixture()
def c2_data_dir():
    return 'data/c2'


@pytest.fixture()
def json_good_rep():
    return '{"scan_id": "dd014af5ed6b38d9130e3f466f850e46d21b951199d53a18ef29ee9341614eaf-1516224457", "resource": "www.google.com", "url": "http://www.google.com/", "response_code": 1, "scan_date": "2018-01-17 21:27:37", "permalink": "https://www.virustotal.com/url/dd014af5ed6b38d9130e3f466f850e46d21b951199d53a18ef29ee9341614eaf/analysis/1516224457/", "verbose_msg": "Scan finished, scan information embedded in this object", "filescan_id": null, "positives": 0, "total": 66, "scans": {"CLEAN MX": {"detected": false, "result": "clean site"}, "DNS8": {"detected": false, "result": "clean site"}, "VX Vault": {"detected": false, "result": "clean site"}, "ZDB Zeus": {"detected": false, "result": "clean site"}, "Tencent": {"detected": false, "result": "clean site"}, "AutoShun": {"detected": false, "result": "unrated site"}, "Netcraft": {"detected": false, "result": "unrated site"}, "PhishLabs": {"detected": false, "result": "unrated site"}, "Zerofox": {"detected": false, "result": "clean site"}, "K7AntiVirus": {"detected": false, "result": "clean site"}, "Virusdie External Site Scan": {"detected": false, "result": "clean site"}, "Spamhaus": {"detected": false, "result": "clean site"}, "Quttera": {"detected": false, "result": "clean site"}, "AegisLab WebGuard": {"detected": false, "result": "clean site"}, "MalwareDomainList": {"detected": false, "result": "clean site", "detail": "http://www.malwaredomainlist.com/mdl.php?search=www.google.com"}, "ZeusTracker": {"detected": false, "result": "clean site", "detail": "https://zeustracker.abuse.ch/monitor.php?host=www.google.com"}, "zvelo": {"detected": false, "result": "clean site"}, "Google Safebrowsing": {"detected": false, "result": "clean site"}, "Kaspersky": {"detected": false, "result": "clean site"}, "BitDefender": {"detected": false, "result": "clean site"}, "Dr.Web": {"detected": false, "result": "clean site"}, "Certly": {"detected": false, "result": "clean site"}, "G-Data": {"detected": false, "result": "clean site"}, "OpenPhish": {"detected": false, "result": "clean site"}, "Malware Domain Blocklist": {"detected": false, "result": "clean site"}, "MalwarePatrol": {"detected": false, "result": "clean site"}, "Webutation": {"detected": false, "result": "clean site"}, "Trustwave": {"detected": false, "result": "clean site"}, "Web Security Guard": {"detected": false, "result": "clean site"}, "CyRadar": {"detected": false, "result": "clean site"}, "desenmascara.me": {"detected": false, "result": "clean site"}, "ADMINUSLabs": {"detected": false, "result": "clean site"}, "Malwarebytes hpHosts": {"detected": false, "result": "clean site"}, "Opera": {"detected": false, "result": "clean site"}, "AlienVault": {"detected": false, "result": "clean site"}, "Emsisoft": {"detected": false, "result": "clean site"}, "Malc0de Database": {"detected": false, "result": "clean site", "detail": "http://malc0de.com/database/index.php?search=www.google.com"}, "malwares.com URL checker": {"detected": false, "result": "clean site"}, "Phishtank": {"detected": false, "result": "clean site"}, "Malwared": {"detected": false, "result": "clean site"}, "Avira": {"detected": false, "result": "clean site"}, "Baidu-International": {"detected": false, "result": "clean site"}, "CyberCrime": {"detected": false, "result": "clean site"}, "Antiy-AVL": {"detected": false, "result": "clean site"}, "Forcepoint ThreatSeeker": {"detected": false, "result": "clean site"}, "FraudSense": {"detected": false, "result": "clean site"}, "Comodo Site Inspector": {"detected": false, "result": "clean site"}, "Malekal": {"detected": false, "result": "clean site"}, "ESET": {"detected": false, "result": "clean site"}, "Sophos": {"detected": false, "result": "unrated site"}, "Yandex Safebrowsing": {"detected": false, "result": "clean site", "detail": "http://yandex.com/infected?l10n=en&url=http://www.google.com/"}, "SecureBrain": {"detected": false, "result": "clean site"}, "Nucleon": {"detected": false, "result": "clean site"}, "Sucuri SiteCheck": {"detected": false, "result": "clean site"}, "Blueliv": {"detected": false, "result": "clean site"}, "ZCloudsec": {"detected": false, "result": "clean site"}, "SCUMWARE.org": {"detected": false, "result": "clean site"}, "ThreatHive": {"detected": false, "result": "clean site"}, "FraudScore": {"detected": false, "result": "clean site"}, "Rising": {"detected": false, "result": "clean site"}, "URLQuery": {"detected": false, "result": "clean site"}, "StopBadware": {"detected": false, "result": "unrated site"}, "Fortinet": {"detected": false, "result": "clean site"}, "ZeroCERT": {"detected": false, "result": "clean site"}, "Spam404": {"detected": false, "result": "clean site"}, "securolytics": {"detected": false, "result": "clean site"}}}'

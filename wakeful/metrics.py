"""
A module to calculate metrics based on the DNS and connection logs produced
by the Bro Network Security Monitor.
"""
import re
import math
import json
import numpy as np
import whois # pip lists it as python-whois
import datetime
import dateutil.parser
from . import ip_address_regex


def calc_pcr(df, src_bytes_col='orig_bytes', dest_bytes_col='resp_bytes'):
    """
    Calculate the producer-consumer ratio for network connections.  The PCR is ratio
    is -1.0 for consumers and 1.0 for producers.  PCR is independent from the speed
    of the network.
    :param df: pandas dataframe
    :param src_bytes_col: name of column with the number of bytes sent by the source
    :param dest_bytes_col: column name with the number of bytes sent by the destination
    :return: pandas series containing the pcr values
    """
    try:
        return (df[src_bytes_col] - df[dest_bytes_col]) / (df[src_bytes_col] + df[dest_bytes_col])
    except ValueError as e:
        fmt = f"Expected '{src_bytes_col}' and '{dest_bytes_col}' to be in dataframe, not {df.columns}."
        print(fmt, file=sys.stderr)
        raise e


def calc_entropy(data, base=2):
    """
    Calculate the entropy of data. Using documentation from
    scipy.stats.entropy as the basis for this code
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html).
    :param data: Measure the entropy of this object
    :return: Calculated entropy value
    """
    if not data:
        return 0

    # calculate frequency list
    chars = set(data)
    frequencies =  [float(data.count(ch)) / len(data) for ch in chars]

    # calculate shannon entropy
    H = -sum([freq  * math.log(freq) / math.log(base) for freq in frequencies ])
    return H


def calc_url_reputation(vt_json):
    """
    Calculate the reputation of a URL based on the VirusTotal.com JSON data.
    the JSON response for this URL from VirusTotal.com.
    :param vt_json: VirusTotal.com JSON response to be processed
    :return: Reputation of the URL where 0.0 is best and 1.0 is worst.
    """
    url_rep = json.loads(vt_json)
    positives = url_rep.get('positives')
    positives = int(positives) if positives else 0
    total = url_rep.get('total')
    total = int(total) if total else 0
    return positives / total if bool(total) else np.nan


def is_new_url(url, num_days=100, set_current_date=None):
    """
    Classify whether a web site has been created recently or not.
    If the URL has both a creation date and a domain name then a
    boolean value will be returned.
    :param url: URL to resolve using DNS
    :param num_days: If a URL was created less than this many days ago, then
    consider it to be new.
    :param use_as_current_date: Provide an ISO formatted string (ISO8601) to
    reset the current date, otherwise today is used.  Obtain with
    datetime.isoformat() that returns a value such as '2014-06-28T12:01:00'.
    :return: True if the URL creation date is less than num_days old,
    otherwise False
    """
    url_info = whois.whois(url)
    if not(bool(url_info.domain) and bool(url_info.creation_date)):
        return np.nan
    if set_current_date:
        comparison_date = dateutil.parser.parse(set_current_date)
    else:
        comparison_date = datetime.today()
    delta = comparison_date - url_info.creation_date
    return delta < datetime.timedelta(days=num_days)


def is_ipv4(ip_address):
    """
    Predicate recognizes IPv4 'dotted quads' addresses.
    :param ip_address: IP address string
    :return: True if it is a valid IPv4 address, otherwise False
    """
    m = re.match(ip_address_regex.ipv4_address, ip_address)
    return m != None


def is_ipv6(ip_address):
    """
    Predicate recognizes IPv6 'dotted quads' addresses.
    :param ip_address: IP address string
    :return: True if it is a valid IPv6 address, otherwise False
    """
    m = re.match(ip_address_regex.ipv6_address, ip_address)
    return m != None


def calc_query_length(query):
    """
    Calculate the length of DNS query.
    :param query: query string
    :return: Length of query string
    """
    return len(query)


def calc_answer_length(answer):
    """
    Calculate the length of DNS answer to a query.
    :param query: answer string
    :return: Length of answer string
    """
    return len(answer)

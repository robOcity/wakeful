"""
A module to calculate metrics based on the DNS and connection logs produced
by the Bro Network Security Monitor.
"""
import math

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
    return (df[src_bytes_col] - df[dest_bytes_col]) / (df[src_bytes_col] + df[dest_bytes_col])


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

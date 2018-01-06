from bat.log_to_dataframe import LogToDataFrame


def bro_to_df(file_path):
    return LogToDataFrame(file_path)


if __name__ == '__main__':

    bro_log_path = '../data/dns.06:00:00-07:00:00.log'
    bro_log_df = LogToDataFrame(bro_log_path)
    print(bro_log_df.head(3))

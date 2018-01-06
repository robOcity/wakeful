from bat.log_to_dataframe import LogToDataFrame


def bro_to_df(file_path):
    return LogToDataFrame(file_path)


if __name__ == '__main__':

    bro_log_path = '../data/dns.06:00:00-07:00:00.log'
    bro_log_df = bro_to_df(bro_log_path)
    print(bro_log_df.shape)
    print(type(bro_log_df))
    print(bro_log_df.columns)

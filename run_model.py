from wakeful import pipelining

def main():
    data_dir = './data/'
    key = 'iodine-forwarded-2017-12-31-dns-train'
    pipelining.modeling_pipeline(key, data_dir)

if __name__ == '__main__':
    main()

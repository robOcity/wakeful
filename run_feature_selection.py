import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, Imputer
from wakeful import log_munger, pipelining, preprocessing


def get_feature_importance(X, y):
    extree = ExtraTreesClassifier()
    extree.fit(X, y)
    return X, extree.feature_importances_

def feature_selection_pipeline(train_df=None, test_df=None, fig_dir=None, fig_file=None, fig_title=None):

    # get features and labels
    X_train, y_train = preprocessing.split_X_y(train_df)

    # create the transformers and estimators
    fillna = Imputer(strategy='median')
    scaler = StandardScaler()
    extree = ExtraTreesClassifier()

    pipe = Pipeline(steps=[
        ('fillna', fillna),
        ('scaler', scaler),
        ('extree', extree),
    ])

    # fit the pipe start to finish
    pipe.fit(X_train, y_train)

    # plot feature 
    column_names = X_train.columns.values
    labels = ['feature', 'importance']
    data = [(name, value) for name, value in zip(column_names, extree.feature_importances_)]
    plotting_df = pd.DataFrame.from_records(data, columns=labels)
    print(plotting_df)
    print(len(column_names), len(extree.feature_importances_))
    plot_feature_importance(column_names, extree.feature_importances_)


def plot_feature_importance(names, values):
    indices = np.argsort(-values)[::-1]
    f, ax = plt.subplots(1, 1, figsize=(7, 5))
    pos = np.arange(len(names)) + 0.5
    plt.barh(pos, values[indices], align='center')
    plt.title("Feature Importance")
    plt.xlabel("Explained Variance")
    plt.ylabel("Feature")
    plt.yticks(pos, names[indices])
    plt.grid(True)
    plt.tight_layout(pad=0.9)
    plt.savefig("./plots/feature_importances")

def plot_bar_chart(plotting_df):
    f, ax = plt.subplots(1, 1, figsize=(7, 5))
    plt.title('Feature Selection')
    ax.set_xlim([0, 1])
    # sns.set_style("whitegrid", { 'axes.edgecolor': '.8', 'text.color': '.15', 'xtick.color': '.15',})
    sns.barplot(x='importance', 
                y='feature', 
                data=plotting_df, 
                palette=sns.color_palette('BuGn_r'))
    # ax.set_xlabel('Importance', fontsize=12)
    # ax.set_ylabel('Feature', fontsize=12)
    sns.despine(offset=10)
    with plt.style.context('seaborn-poster'):
        plt.tight_layout(pad=0.8)
        plt.show()


def LeaveOneOut(data1, data2, columnName, useLOO=False):
    # cite: inspired by https://www.kaggle.com/shrutigodbole15792/feature-selection
    grpOutcomes = data1.groupby(columnName).mean().reset_index()
    outcomes = data2['outcome'].values
    x = pd.merge(data2[[columnName, 'outcome']], grpOutcomes,
                 suffixes=('x_', ''),
                 how='left',
                 on=columnName,
                 left_index=True)['outcome']
    if(useLOO):
        x = ((x*x.shape[0])-outcomes)/(x.shape[0]-1)
    return x.fillna(x.mean())


def fig_title(h5_key):
    tokens = [tok.capitalize() for tok in h5_key.split('_')]
    return f'Malware: {tokens[0]} -- Log Data Analyzed: {tokens[-2]}'



def fig_name(h5_key):
    tokens = [tok.capitalize() for tok in h5_key.split('_')]
    return f'{tokens[0]}_{tokens[-2]}.png'


if __name__ == '__main__':

    log_types = ['dns', 'conn']

    all_data = dict([
        ('dnscat2_2017_12_31_conn_test', 'dnscat2_2017_12_31_conn_train'),
        ('dnscat2_2017_12_31_dns_test', 'dnscat2_2017_12_31_dns_train'),
        ('iodine_forwarded_2017_12_31_conn_test', 'iodine_forwarded_2017_12_31_conn_train'),
        ('iodine_forwarded_2017_12_31_dns_test', 'iodine_forwarded_2017_12_31_dns_train'),
        ('iodine_raw_2017_12_31_conn_test', 'iodine_raw_2017_12_31_conn_train'),
        ('iodine_raw_2017_12_31_dns_test', 'iodine_raw_2017_12_31_dns_train'),
    ])

    half_data = dict([
        ('dnscat2_2017_12_31_dns_test', 'dnscat2_2017_12_31_dns_train'),
        ('iodine_forwarded_2017_12_31_conn_test', 'iodine_forwarded_2017_12_31_conn_train'),
        ('iodine_raw_2017_12_31_conn_test', 'iodine_raw_2017_12_31_conn_train'),
    ])

    small_data = dict([
        ('iodine_raw_2017_12_31_dns_test', 'iodine_raw_2017_12_31_dns_train'),
    ])

    data_dir = 'data/'
    fig_dir = 'plots/'
    data = half_data
    train_dfs, test_dfs = [], []

    for test_key, train_key in data.items():
        train_dfs.append(log_munger.hdf5_to_df(train_key, data_dir))
        test_dfs.append(log_munger.hdf5_to_df(test_key, data_dir))

    train_df = pd.concat(train_dfs)
    test_df = pd.concat(test_dfs)

    print(train_df.head())
    print(train_df.info())

    feature_selection_pipeline(train_df=train_df, test_df=None, fig_dir='plots', fig_file='extree', fig_title='Feature Importances')

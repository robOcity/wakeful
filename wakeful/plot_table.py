import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns; sns.set()

# TODO - pass in data
data = [[ 0.916085777907, 0.915616944701, 0.916589884099, 0.196632551623,  0.211288112616, 0.2308009012],
        [ 0.928623293639, 0.954780749378, 0.901791421807, 0.803367448377,  0.211257749068, 0.230859036967],
        [ 0.414221098683, 0.808514540629, 0.644771511243, 0.500000000000,  0.627595992448, 0.802474677585]]

columns = ('Conn Log\nDNSCat2', 'Conn Log\nIodine', 'Conn Log\nIodine-Raw',
           'DNS Log\nDNSCat2', 'DNS Log\nIodine', 'DNS Log\nIodine-Raw')
rows = ['Boosting', 'Random\nForest', 'Logistic\nRegression']

df = pd.DataFrame(data, columns=columns, index=rows).T

sns.set()

with sns.plotting_context():
    f, ax = plt.subplots(1, 1, figsize=(6, 5))
    plt.subplots_adjust(hspace=.15)
    ax.set_xlabel('Data Source\nMalware', fontsize=18)
    ax.set_ylabel('Model', fontsize=18)
    plt.title('Model Evaluation Results\nArea Under Curve', fontsize=18)
    ax = sns.heatmap(df,
                     annot=True,
                     fmt='.2f',
                     cmap=sns.color_palette('BuGn'))

    plt.tight_layout(pad=0.9)
    plt.show()

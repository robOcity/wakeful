import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns; sns.set()

sns.set(rc={"figure.figsize": (6, 6)})

data = [[ 0.916085777907, 0.915616944701, 0.916589884099, 0.196632551623,  0.211288112616, 0.2308009012],
        [ 0.928623293639, 0.954780749378, 0.901791421807, 0.803367448377,  0.211257749068, 0.230859036967],
        [ 0.414221098683, 0.808514540629, 0.644771511243, 0.500000000000,  0.627595992448, 0.802474677585]]

columns = ('Conn-DNSCat2', 'Conn-Iodine', 'Conn-Iodine-Raw', 'DNS-DNSCat2', 'DNS-Iodine', 'DNS-Iodine-Raw')
rows = ['Boosting', 'Random Forest', 'Logistic\nRegression']

df = pd.DataFrame(data, columns=columns, index=rows)

ax = sns.heatmap(df,
                 annot=True,
                 linewidths=0.5,
                 fmt='.2f',
                 cmap=sns.color_palette("Blues"))

plt.title('Model Evaluation Results\nArea Under Curve', fontsize=18)
plt.show()

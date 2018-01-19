## make imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style="ticks", color_codes=True)

## create some data that is too big for pairplot
d = 5
n = 1000
row_labels = np.array(['r-%s'%r for r in range(n)])
col_labels = np.array(['c-%s'%c for c in range(d)])
x = np.random.normal(0,5,n*d).reshape(n,d)

## subsample the observations (say 500)
subsample_size = 500
rand_inds = np.random.randint(0,n,subsample_size)
x_subsample = x[rand_inds,:]
subsample_rows = row_labels[rand_inds]

## turn your array into a dataframe
df =  pd.DataFrame(data=x_subsample,
                   index=subsample_rows,
                   columns=col_labels)

g = sns.pairplot(df)
plt.show()

import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns


to_compare = ['320', '320without', '1025']

# dfs = [pd.read_pickle('eval-{}.pd'.format(name)) for name in to_compare]
dfs = [pd.read_pickle('eval_v3-{}.pd'.format(name)) for name in to_compare]

plt.subplot(221)
plt.title('Direct hits')
sns.barplot(to_compare, [df['correct'].mean() for df in dfs])
plt.ylim([0, 1])


plt.subplot(222)
plt.title('Genus hits')
sns.barplot(to_compare, [df['genus_correct'].mean() for df in dfs])
plt.ylim([0, 1])


plt.subplot(223)
plt.title('Unknown plant')
sns.barplot(to_compare, [1 - df['class_known'].mean() for df in dfs])
plt.ylim([0, 1])


plt.subplot(224)
plt.title('Hit if known')
sns.barplot(to_compare, [df['correct'].mean() / df['class_known'].mean() for df in dfs])
plt.ylim([0, 1])

plt.show()

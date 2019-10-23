import numpy as np

n = 1000 * 1000

np.random.seed(1)
x1 = np.random.randint(2, size=n)
# x2 = np.random.randint(2, size=10000)
np.random.seed(2)
y1 = np.random.randint(2, size=n)

from scipy.stats import chisquare
b = np.bincount(x1 + y1 * 2) # = contingency
chisquare(b, ddof=1)
#Power_divergenceResult(statistic=7.0496, pvalue=0.029457698319656538)

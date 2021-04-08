
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd

# READ
df_train = pd.read_csv('./data/training.csv', header=None)
df_train.columns = ['Open', 'High', 'Low', 'Close']


# plt.plot(df_train['Open'])
# plt.show()

# # 
# def piecewise_linear(x, x0, y0, k1, k2):
#     return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

# p , e = optimize.curve_fit(piecewise_linear, df_train.index, df_train['Open'])

# plt.plot(df_train.index, df_train['Open'], "o")
# plt.plot(df_train.index, piecewise_linear(df_train.index, *p))
# plt.show()

import numpy as np
import numpy.polynomial.polynomial as npoly
from scipy import optimize
import matplotlib.pyplot as plt
np.random.seed(2017)

def f(breakpoints, x, y, fcache):
    breakpoints = tuple(map(int, sorted(breakpoints)))
    if breakpoints not in fcache:
        total_error = 0
        for f, xi, yi in find_best_piecewise_polynomial(breakpoints, x, y):
            total_error += ((f(xi) - yi)**2).sum()
        fcache[breakpoints] = total_error
    # print('{} --> {}'.format(breakpoints, fcache[breakpoints]))
    return fcache[breakpoints]

def find_best_piecewise_polynomial(breakpoints, x, y):
    breakpoints = tuple(map(int, sorted(breakpoints)))
    xs = np.split(x, breakpoints)
    ys = np.split(y, breakpoints)
    result = []
    for xi, yi in zip(xs, ys):
        if len(xi) < 2: continue
        coefs = npoly.polyfit(xi, yi, 1)
        f = npoly.Polynomial(coefs)
        result.append([f, xi, yi])
    return result

x = df_train.index[-21:]
y = df_train['Open'][-21:]



num_breakpoints = 3
breakpoints = optimize.brute(
    f, [slice(1, len(x), 1)]*num_breakpoints, args=(x, y, {}), finish=None)

plt.scatter(x, y, c='blue', s=50)
for f, xi, yi in find_best_piecewise_polynomial(breakpoints, x, y):
    x_interval = np.array([xi.min(), xi.max()])
    print('y = {:35s}, if x in [{}, {}]'.format(str(f), *x_interval))
    plt.plot(x_interval, f(x_interval), 'ro-')


plt.show()

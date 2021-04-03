import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./data/training.csv',header=None)
# open-high-low-close


print(df[3])

plt.plot(df[3])
plt.show()
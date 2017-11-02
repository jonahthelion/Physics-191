import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# df = pd.read_csv('../data/DPPH.csv', skiprows=range(15))
# plt.plot(df.CH1.values)
# plt.show()

acvals = [20,30,40,50,60, 70, 80, 90, 100, 10, 5]
maxes = np.array([3.34,3.35,3.37,3.39,3.4, 3.42, 3.43, 3.45, 3.47, 3.32, 3.315])
mins = np.array([3.28,3.26,3.25,3.23,3.21, 3.2, 3.18, 3.17, 3.15, 3.29, 3.3])

p1 = np.polyfit(acvals, maxes-mins, 1)
x_show = np.array([0,100])
plt.plot(x_show, p1[0]*x_show + p1[1])

plt.plot(acvals, maxes-mins, 'o')
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.patches as mpatches

# Matplotlib settings
plt.style.use('ggplot')
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams.update({'font.size': 18})
plt.ion()



def slope_func(x, m):
    return m*x

def make_calib_plot(file_name):

    acvals = [20,30,40,50,60, 70, 80, 90, 100, 10, 5]
    maxes = np.array([3.34,3.35,3.37,3.39,3.4, 3.42, 3.43, 3.45, 3.47, 3.32, 3.315])
    mins = np.array([3.28,3.26,3.25,3.23,3.21, 3.2, 3.18, 3.17, 3.15, 3.29, 3.3])

    p1,_ = curve_fit(slope_func, acvals, maxes-mins)
    print p1
    x_show = np.array([0,100])

    fig = plt.figure(figsize=(10,6))
    plt.plot(x_show, p1[0]*x_show, 'k')

    plt.plot(acvals, maxes-mins, 'o')

    plt.ylabel(r'$2|\widetilde{B}|$ (kG)')
    plt.xlabel(r'$C$ $(|\widetilde{B}|$ knob value)')

    plt.axvline(1, linestyle='--', color='b')

    plt.legend(handles = [mpatches.Patch(color='k', label='Calibration Fit')])
    plt.annotate(r'$2|\widetilde{B}| = $' + str(round(p1[0],5)) + r'$\times C$', xy=(.1,.7), xycoords='axes fraction')

    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()

make_calib_plot('../images/dpph_calib.pdf')


# df = pd.read_csv('../data/DPPH.csv', skiprows=range(15))

# fig = plt.figure(figsize=(10,6))
# plt.plot(df.CH1.values)

# ax = plt.gca()
# ax1 = ax.twinx()

# plt.plot(df.CH2.values, 'r--')

# plt.show()




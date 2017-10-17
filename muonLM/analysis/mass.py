import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from scipy.optimize import curve_fit
from tqdm import tqdm
import matplotlib.gridspec as gridspec
import seaborn
import matplotlib as mpl
import pandas as pd
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as mpatches
from scipy.optimize import curve_fit
from matplotlib import gridspec

# Random seed
np.random.seed(42)

# Matplotlib settings
plt.rcParams['font.serif'] = 'Ubuntu'

def read_data(file):
    data = []
    with open(file) as reader:
        for line in reader:
            data.append(map(float,line.rstrip().split('\t')))
    data = np.array(data)

    if 'Lifetime' in file:
        columns = ['clock', 'life']
    else:
        columns = ['clock', 'e', 'mu', 'life']

    data = pd.DataFrame(data, columns=columns)
    return data


df = read_data('../data/2017_09_21_Energy_Calib.txt')

fig = plt.figure(figsize=(10,6))
n,bins,paint = plt.hist(df.e.values, bins=np.arange(min(df.e),max(df.e)+.02,.02))
middles = (bins[1:] + bins[:-1]) / 2.
PROP = middles[np.argmax(n)]
ax = plt.gca()
ax.axvline(PROP, linestyle='--', color='k')
plt.xlabel('Voltage (mV)')
plt.ylabel('Counts')
plt.title('Calibration: Voltage to $\mu$ Energy')
plt.annotate('Calibration: ' + str(round(PROP, 3)), xy=(0.75, 0.5), xycoords = 'axes fraction')
plt.tight_layout()

plt.savefig('../plots/mass_calib.pdf')
plt.close(fig)


df = read_data('../data/2017_09_21_Mass2.txt')
fig = plt.figure(figsize=(10,6))
n, bins, paint = plt.hist(df.e.values, bins=np.arange(min(df.e),max(df.e) + .04,.04))
middles = (bins[1:] + bins[:-1]) / 2.
greater_than_ixes = np.array([ix for ix in range(len(n)) if n[ix] > 1])
chosen_ix = max(greater_than_ixes)
TOP = middles[chosen_ix]
ax = plt.gca()
ax.set_yscale('log')
plt.xlabel('Voltage (mV)')
plt.ylabel('Counts')
ax.axvline(TOP, color='k', linestyle='--')
plt.title('e- Voltages')
plt.annotate('Maximum $e-$ Voltage: ' + str(round(TOP, 3)) + ' mV', xy=(0.5, 0.8), xycoords = 'axes fraction')
plt.tight_layout()

plt.savefig('../plots/mass2.pdf')
plt.close(fig)

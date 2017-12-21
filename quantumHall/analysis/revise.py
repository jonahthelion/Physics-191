import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.patches as mpatches
from scipy.optimize import curve_fit
from tqdm import tqdm
import pickle
import itertools
from utils import *

# Matplotlib settings
plt.style.use('ggplot')
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams.update({'font.size': 18})
plt.ion()

data_dir = '../data/'
plot_dir = '../plots/'

ftemps = get_ftemps()
# df = get_data(os.path.join(data_dir, '10_19_2017_Run3_Jonah_Katie.txt'), os.path.join(plot_dir, ''))
# p_classical = classical_plot(df, os.path.join(plot_dir, 'classical2.jpg'))

info = {}
for f in ftemps:
    fileout = os.path.join(plot_dir, 'basic_' + str(ftemps[f]['temp'][0]) + '.pdf')
    info[f] = get_data(os.path.join(data_dir, f))

# #### Classical Plots
# slopes = [info[f][2][0] for f in info]
# constants = [info[f][3] for f in info]
# f = '10_17_2017_Run1_Jonah_Katie.txt'
# get_classical(info[f], slopes, constants, os.path.join(plot_dir, 'classical_' + str(ftemps[f]['temp'][0]) + '.pdf'))

# ##### Quantum Plots
# actual_minimas = []
# for f in info:
#     fileout = os.path.join(plot_dir, 'quant_' + str(ftemps[f]['temp'][0]) + '.pdf')
#     actual_minimas.append(get_quantum(info[f], fileout))
# plot_actual_minimas(info, actual_minimas, os.path.join(plot_dir, 'minima.pdf'))

# for f in info:
#     fileout = os.path.join(plot_dir, 'int_' + str(ftemps[f]['temp'][0]) + '.pdf')
#     get_int(info[f], fileout)

### find temperature dependence
means = []
for m_star in np.linspace(1.5,1.6, 1):
    means.append(get_temp(info, os.path.join(plot_dir, 'temp'), ftemps, m_star*9.1e-31))
print np.linspace(.08, 1.4, 100)[np.argmin(abs(np.array(means) - 1))]


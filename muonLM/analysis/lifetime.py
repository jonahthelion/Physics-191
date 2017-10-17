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
import itertools
import multiprocessing as mp

# Random seed
np.random.seed(42)

# Matplotlib settings
plt.rcParams['font.serif'] = 'Ubuntu'

def weight_array(ar, weights):
     zipped = zip(ar, weights)
     weighted = []
     for i in zipped:
         for j in range(i[1]):
             weighted.append(i[0])
     return weighted

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

def exp_fit(x, lamb, k, ped):
    return k*lamb*pow(np.e, -x*lamb) + ped

def two_exp(x, lamb1, lamb2, k, ped):
    return 3.52*k*lamb1*pow(np.e, -x*lamb1) + k*lamb2*pow(np.e, -x*lamb2)+ ped

def lifetime_plot(df, col_name, save_name):

    min_cuts = np.linspace(1.0, 1.5, 5)
    max_cuts = np.linspace(5., 6., 5)
    bin_mults = np.linspace(50, 60, 2)

    lifetimes = []

    def get_life(row):
        min_cut,max_cut,bin_mult = row
        fig1 = plt.figure()
        n, bins, patches = plt.hist(df[col_name].values, bins=np.arange(0+.001,20+.001,0.002*bin_mult), color='b')
        middles = (bins[1:] + bins[:-1]) / 2.
        kept_ixes = [ix for ix in range(len(middles)) if min_cut<=middles[ix]<=max_cut]
        uncert = np.sqrt(n)[kept_ixes]
        popt,pcov = curve_fit(exp_fit ,middles[kept_ixes] , n[kept_ixes], sigma=uncert)    
        lifetime = 1./popt[0]   
        plt.close(fig1)
        print 'RAT:', popt[2] / float(sum(n[kept_ixes])), 1/popt[0]
        return lifetime, pcov[0][0]

    lifetimes = np.array(map(get_life, itertools.product(min_cuts, max_cuts, bin_mults)))
    return None,None,None


    # fig = plt.figure(figsize=(10,6))
    # lifetime = np.percentile(weight_array(lifetimes[:,0], 1./lifetimes[:,1]), 50.)
    # lerr = round(lifetime - np.percentile(weight_array(lifetimes[:,0], 1./lifetimes[:,1]), 25.), 3)
    # rerr = round(np.percentile(weight_array(lifetimes[:,0], 1./lifetimes[:,1]), 75.) - lifetime, 3)
    # ax = plt.gca()
    # plt.hist(lifetimes, bins=20)
    # ax.add_patch(mpatches.Rectangle((lifetime - lerr, 0), lerr + rerr, ax.get_ylim()[1], alpha=.2, color='goldenrod'))
    # plt.axvline(lifetime, color='k', linestyle='--')
    # plt.xlabel(r'Lifetime ($\mu s$)')
    # plt.ylabel('Counts')
    # plt.title('Lifetime Systematic Uncertainty')
    # plt.annotate('Lifetime: ' + str(round(lifetime, 2)) + r'$^{+' + str(rerr) + r'}_{-' + str(lerr) + r'}$' + r' $ \mu$s', xy=(0.75, 0.5), xycoords = 'axes fraction')
    # plt.savefig(save_name[:-4] + '_syst.pdf')
    # plt.close(fig)

    # # lifetimes = []
    # # for check in range(20):
    # #     fig_2 = plt.figure()
    # #     n, bins, patches = plt.hist(np.random.choice(df[col_name].values, int(.5 * len(df)),replace=False), bins=np.arange(0+.001,20+.001,0.002*50), color='b')
    # #     middles = (bins[1:] + bins[:-1]) / 2.
    # #     kept_ixes = [ix for ix in range(len(middles)) if min_cut<=middles[ix]<=max_cut]
    # #     uncert = np.sqrt(n)[kept_ixes]
    # #     popt,pcov = curve_fit(exp_fit ,middles[kept_ixes] , n[kept_ixes], sigma=uncert) 
    # #     lifetimes.append(popt[0])
    # #     plt.close(fig_2)
    # # err = np.std(1./np.array(lifetimes))

    # fig = plt.figure(figsize=(10,6))
    # G = gridspec.GridSpec(3,1)
    # ax1 = plt.subplot(G[:2, :])

    # min_cut,max_cut,bin_mult = (1.0, 15.0, 50.0)
    # n, bins, patches = plt.hist(df[col_name].values, bins=np.arange(0+.001,20+.001,0.002*bin_mult), color='b')
    # middles = (bins[1:] + bins[:-1]) / 2.
    # kept_ixes = [ix for ix in range(len(middles)) if min_cut<=middles[ix]<=max_cut]
    # uncert = np.sqrt(n)[kept_ixes]
    # popt,pcov = curve_fit(exp_fit ,middles[kept_ixes] , n[kept_ixes], sigma=uncert)    
        


    # preds = np.array([exp_fit(mi, popt[0], popt[1], popt[2]) for mi in middles])
    # plt.plot(middles, preds, color='g')

    # plt.axvline(min_cut, linestyle='--', color='k')
    # plt.axvline(max_cut, linestyle='--', color='k')


    # ax = fig.gca()
    # plt.tick_params(
    # axis='x',          # changes apply to the x-axis
    # which='both',      # both major and minor ticks are affected
    # bottom='off',      # ticks along the bottom edge are off
    # top='off',         # ticks along the top edge are off
    # labelbottom='off') # labels along the bottom edge are off

    # plt.legend(handles=[mpatches.Patch(color='b', label='Events'),
    #                     mpatches.Patch(color='k', label='Cuts'),
    #                     mpatches.Patch(color='g', label='Fit')])

    # minorLocator = MultipleLocator(1.0)
    # ax.yaxis.set_minor_locator(minorLocator)
    # ax.xaxis.set_minor_locator(minorLocator)
    # ax.grid(which = 'minor')

    # plt.ylabel(r'Counts')
    # plt.title(r'$\mu$ Lifetime Calculation')
    # plt.yscale('log')
    # plt.annotate('Lifetime: ' + str(round(lifetime, 2)) + r'$^{+' + str(rerr) + r'}_{-' + str(lerr) + r'}$' + r' $ \mu$s', xy=(0.75, 0.5), xycoords = 'axes fraction')


    # ax1 = plt.subplot(G[2, :], sharex=ax)
    # ax1.bar(middles[kept_ixes], preds[kept_ixes] - n[kept_ixes], bins[1]-bins[0], 0, align='center', yerr=uncert)
    # plt.ylabel('Fit - Data (Counts)')
    # ax1.add_patch(mpatches.Rectangle((ax1.get_xlim()[0],-np.std(preds[kept_ixes] - n[kept_ixes])), ax1.get_xlim()[1] - ax1.get_xlim()[0], 2*np.std(preds[kept_ixes] - n[kept_ixes]), angle=0.0, color='darkgoldenrod', alpha=.2))
    # # minorLocator = MultipleLocator(1.0)
    # # ax1.yaxis.set_minor_locator(minorLocator)
    # # ax1.xaxis.set_minor_locator(minorLocator)
    # ax1.grid(which = 'minor')
    # plt.legend(handles=[mpatches.Patch(color='darkgoldenrod',alpha=.2, label='RMS: ' + str(round(np.std(preds[kept_ixes] - n[kept_ixes]), 2)) + ' counts')])

    # plt.xlabel(r'Decay Time ($\mu$s)')
    # plt.tight_layout()
    # plt.savefig(save_name)
    # plt.close(fig)

    # return round(lifetime, 2), lerr, rerr, lifetimes






files = ['../data/2017_09_21_Mass.txt', '../data/2017_09_21_Mass2.txt']
all_lifes = []
for file_ix, file in enumerate(files):
    df = read_data(file)
    add_in = [df.life.values[ix] for ix in range(len(df)) if df.e.values[ix] > .065]
    all_lifes.extend(add_in)

bin_mult = 50
min_cut = 1.1
max_cut = 15.0
fig = plt.figure(figsize=(10,6))
n, bins, patches = plt.hist(all_lifes, bins=np.arange(0+.001,20+.001,0.002*bin_mult), color='b')


min_cuts = np.linspace(1.0, 1.5, 15)
max_cuts = np.linspace(14., 17.5, 15)
bin_mults = np.linspace(10, 60, 6).astype(int)

middles = (bins[1:] + bins[:-1]) / 2.
kept_ixes = [ix for ix in range(len(middles)) if min_cut<=middles[ix]<=max_cut]
uncert = np.sqrt(n)[kept_ixes]
popt,pcov = curve_fit(exp_fit ,middles[kept_ixes] , n[kept_ixes], sigma=uncert) 
print 1./popt[0]

lifetimes = []
def get_life(row):
    min_cut,max_cut,bin_mult = row
    fig1 = plt.figure()
    n, bins, patches = plt.hist(all_lifes, bins=np.arange(0+.001,20+.001,0.002*bin_mult), color='b')
    middles = (bins[1:] + bins[:-1]) / 2.
    kept_ixes = [ix for ix in range(len(middles)) if min_cut<=middles[ix]<=max_cut]
    uncert = np.sqrt(n)[kept_ixes]
    popt,pcov = curve_fit(exp_fit ,middles[kept_ixes] , n[kept_ixes], sigma=uncert)    
    lifetime = 1./popt[0]   
    plt.close(fig1)
    print 'RAT:', popt[2] / float(sum(n[kept_ixes])), 1/popt[0]
    return lifetime, pcov[0][0]

lifetimes = np.array(map(get_life, itertools.product(min_cuts, max_cuts, bin_mults)))



plt.yscale('log')
plt.show()








# df = read_data('../data/2017_09_14_Muon_Lifetime.txt')
# lifetimes, lerr, rerr = lifetime_plot(df, 'life', '../plots/life0.pdf')




# files = ['../data/2017_09_14_Muon_Lifetime.txt', '../data/2017_09_21_Mass.txt', '../data/2017_09_21_Mass2.txt']
# info = []
# for file_ix, file in enumerate(files):
#     print "NAJFANFAFN:AFJA:AF:"
#     df = read_data(file)
#     life = lifetime_plot(df, 'life', '../plots/life' + str(file_ix) + '.pdf')
#     info.append(life)


# lifes = np.array(lifes)
# errs = np.array(errs)
# comb = np.polyfit(range(len(lifes)),lifes, deg=0, w=1/errs)[0]
# err = .028
# lifes = np.append(lifes, comb)
# errs = np.append(errs, err)

# fig = plt.figure(figsize=(10,6))
# plt.errorbar(range(len(lifes)), lifes, xerr=0.0, yerr=errs, fmt='o')
# plt.xticks(range(len(lifes)), ['Run1', 'Run2', 'Run3', 'Combined'])
# plt.axvline(len(lifes) - 1.5, color='black', linestyle='--')
# plt.ylabel('Lifetime ($\mu s$)')
# plt.title('$\mu$ Lifetime')

# plt.savefig('../plots/LIFE.pdf')
# plt.close(fig)








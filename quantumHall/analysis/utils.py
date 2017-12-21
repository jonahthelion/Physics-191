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
from scipy.signal import argrelextrema

# Matplotlib settings
plt.style.use('ggplot')
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams.update({'font.size': 18})
plt.ion()

def get_data(file, fileout=None):
    df = pd.read_csv(file, skiprows=3, delimiter='\t', header=None, 
                names=['imag', 'lock1', 'lock2', 't'])
    # add sweep up/down
    top_ix = np.argmax(df.imag.values)
    df = df[np.logical_and(df.imag > 0.4, df.imag < df.imag.values[top_ix] - .1)]
    df.loc[df.index < top_ix, 'sweep'] = 1
    df.loc[df.index > top_ix, 'sweep'] = -1
    df.index = range(len(df))

    # add magnetic field
    df['B'] = .1172 * df.imag.values

    # add rho xy
    df['xy'] = df.lock2.values/10.*2e-3/45.5e-9

    # add rho xx
    df['xx'] = df.lock1.values/10.*100e-6/45.5e-9*.35

    if not fileout is None:
        print 'plotting to', fileout
        fig = plt.figure(figsize=(12,11))
        gs = mpl.gridspec.GridSpec(4, 1)

        # rho xx plot
        ax = plt.subplot(gs[0])
        plt.plot(df[df.sweep==1].B, df[df.sweep == 1].xx.values, 'b')
        plt.plot(df[df.sweep==-1].B.values, df[df.sweep == -1].xx.values, 'g')
        plt.xlabel('B (T)')
        plt.ylabel(r'$\rho_{xx}$ $(\Omega$)$')
        plt.legend(handles=[mpatches.Patch(label=r'$\uparrow$', color='b'),
                            mpatches.Patch(label=r'$\downarrow$', color='g')])

        # rho xy plot
        ax = plt.subplot(gs[1])
        plt.plot(df[df.sweep==1].B, df[df.sweep == 1].xy.values, 'b')
        plt.plot(df[df.sweep==-1].B.values, df[df.sweep == -1].xy.values, 'g')
        plt.xlabel('B (T)')
        plt.ylabel(r'$\rho_{xy}$ $(\Omega)$')

    # shift in x plot
    x_shift = .04
    df.loc[df.sweep == 1, 'B'] -= x_shift/2.
    df.loc[df.sweep == -1, 'B'] += x_shift/2.

    if not fileout is None:
        ax = plt.subplot(gs[2])
        plt.plot(df[df.sweep==1].B.values, df[df.sweep == 1].xx.values, 'b')
        plt.plot(df[df.sweep==-1].B.values, df[df.sweep == -1].xx.values, 'g')
        plt.xlabel('B (T)')
        plt.ylabel(r'$\rho_{xx}$ $(\Omega)$')

    # shift in y plot
    small_ixes = np.logical_and(df.B<.4, df.sweep == 1)
    p_shift = np.polyfit(df[small_ixes].B.values, df[small_ixes].xy.values, 1)
    df.loc[:,'xy'] -= p_shift[1]

    # find constant
    constant = np.mean(df[small_ixes].xx.values)

    if not fileout is None:
        ax = plt.subplot(gs[3])
        p = np.polyfit(df[small_ixes].B.values, df[small_ixes].xy.values, 1)
        x_show = np.linspace(0, max(df[small_ixes].B.values), 100)
        plt.plot(df[small_ixes].B.values, df[small_ixes].xy.values, '.')
        plt.plot(x_show, x_show*p[0]+p[1], 'k')
        plt.xlabel('B (T)')
        plt.ylabel(r'$\rho_{xy}$ $(\Omega)$')
        plt.annotate(r'$\rho_{xy}=$ ' + str(round(p[0], 2)) + ' B', xy=(.1,.8), xycoords='axes fraction')

        plt.tight_layout()
        plt.savefig(fileout)
        #plt.show()
        plt.close(fig)

    print 'Sweep Speed:', (df[df.sweep == 1].B.values[-1] - df[df.sweep == 1].B.values[0]) / (df[df.sweep == 1].t.values[-1] - df[df.sweep == 1].t.values[0])

    return df,x_shift,p_shift,constant

def get_ftemps():
    return {#'10_10_2017_4K_Run7_Jonah_Katie.txt': {'temp': (4.0, 4.1, 3.9)},
            '10_17_2017_Run1_Jonah_Katie.txt': {'temp':(3.07,  3.035, 3.104)},
            '10_17_2017_Run2_Jonah_Katie.txt': {'temp':(2.508, 2.454, 2.584)},
            '10_17_2017_Run3_Jonah_Katie.txt': {'temp':(1.947, 1.886, 2.075)},
            '10_19_2017_Run1_Jonah_Katie.txt': {'temp': (1.913, 1.744, 1.978)},
            '10_19_2017_Run2_Jonah_Katie.txt':{'temp':(1.858, 1.793, 1.93)},}


def get_classical(info, slopes, constants, fileout):
    df,x_shift,p_shift,constant = info
    print 'plotting classical to', fileout

    # plot line
    fig = plt.figure(figsize=(10,6))

    small_ixes = np.logical_and(df.B<.4, df.sweep == 1)
    x_show = df[small_ixes].B.values
    y_show = df[small_ixes].xy.values
    plt.plot(x_show, y_show, '.')
    p = np.polyfit(x_show, y_show, 1)
    plt.plot(np.linspace(0,max(x_show), 1000), p[0]*np.linspace(0,max(x_show), 1000)+p[1], 'k')

    plt.xlabel('B (T)')
    plt.ylabel(r'$\rho_{xy}$ ($\Omega$)')
    plt.annotate(r'$\rho_{xy} =$ ' + str(round(np.mean(slopes), 2)) + r'$^{+' + str(round(np.std(slopes), 2)) + r'}' + r'_{-' + str(round(np.std(slopes), 2)) + r'}$' + r' $B$', xy=(.1,.8), xycoords='axes fraction')

    plt.tight_layout()
    plt.savefig(fileout[:-4] + '_0.pdf')
    plt.close(fig)

    # plot constant
    fig = plt.figure(figsize=(10,6))

    small_ixes = np.logical_and(df.B<.4, df.sweep == 1)
    x_show = df[small_ixes].B.values
    y_show = df[small_ixes].xx.values
    plt.plot(x_show, y_show, '.')
    plt.axhline(np.mean(y_show), color='k', linestyle='--')

    plt.xlabel('B (T)')
    plt.ylabel(r'$\rho_{xx}$ ($\Omega$)')
    plt.annotate(r'$\langle \rho_{xx} \rangle=$ ' + str(round(np.mean(constants), 2)) + r'$^{+' + str(round(np.std(constants), 2)) + r'}' + r'_{-' + str(round(np.std(constants), 2)) + r'}$', xy=(.1,.8), xycoords='axes fraction')
    plt.ylim((0, max(y_show)*2))
    plt.tight_layout()
    plt.savefig(fileout[:-4] + '_1.pdf')
    plt.close(fig)

def get_quantum(info, fileout):
    df,x_shift,p_shift,constant = info
    print 'quantum saved to', fileout

    # restrict to up sweep with error from down sweep
    df = df[df.sweep == 1]

    rough_minima = [5.2,3.5, 2.6, 1.8, 1.4, 1.1, .9, .7, .6, .5, .45]
    actual_minima = []
    for r in rough_minima:
        if r > max(df.B):
            actual_minima.append(None)
            continue
        closest = np.argmin(np.abs(df.B.values - r))
        actual_minima.append(np.argmin(df.xx.values[(closest - 50):(closest+50)]) + closest - 50)

    fig = plt.figure(figsize=(10,6))

    plt.plot(df.B, df.xx)
    plt.plot(df.B.values[[act for act in actual_minima if not act is None]], df.xx.values[[act for act in actual_minima if not act is None]], 'b.')

    for ix,val in enumerate(actual_minima):
        if not val is None and not 5<ix<10:
            xy = [df.B.values[val], df.xx.values[val]]
            if ix == 10:
                xy[0] -= .22
                xy[1] += 2
            plt.annotate(str(ix+2), xy=tuple(xy))

    plt.xlabel('B (T)')
    plt.ylabel(r'$\rho_{xx}$ ($\Omega$)')

    plt.tight_layout()
    plt.savefig(fileout)
    plt.close(fig)

    return [df.B.values[act] if not act is None else None for act in actual_minima]

def plot_actual_minimas(info, actual_minimas, fileout):
    fig = plt.figure(figsize=(10,6))

    print 'plotting actual minimas', fileout
    # for actual_minima in actual_minimas:
    #     plt.plot([1./val if not val is None else None for val in actual_minima], '.')

    means = []; stds = [];
    for label in range(11):
        vals = [1./actual_minima[label] for actual_minima in actual_minimas if not actual_minima[label] is None]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    plt.errorbar(np.arange(len(means))+2, means, yerr=stds, fmt='.')

    p,V = np.polyfit(np.arange(len(means))[2:]+2, means[2:], w=1./np.array(stds[2:]),deg=1, cov=True)
    x_show = np.arange(4, 13)
    plt.plot(x_show, p[0]*x_show + p[1], 'k')

    plt.annotate(r'$\frac{1}{B} =$ ' + str(round(p[0], 3)) + r'$^{+' + str(round(.002, 3)) + r'}' + r'_{-' + str(round(.002, 3)) + r'}$'+ r' $L$', xy=(.1,.8), xycoords='axes fraction')

    plt.xlabel(r'Label ($\nu$)')
    plt.ylabel(r'$1/B$ $(T^{-1})$')

    plt.ylim((0, 2))
    plt.tight_layout()
    plt.savefig(fileout)
    plt.close(fig)

def get_int(info, fileout):

    df,x_shift,p_shift,constant = info
    print 'integer saved to', fileout    

    # restrict to up sweep with error from down sweep
    df = df[df.sweep == 1]
    diffs = np.diff(df.xy.values)
    diffs = np.append(diffs, diffs[-1])

    fig = plt.figure(figsize=(10,6))
    gs = mpl.gridspec.GridSpec(2, 1)

    ax = plt.subplot(gs[0])
    plt.plot(df.B, df.xy)

    reses = {}
    for n in range(2, 11):
        val = 6.626e-34 / (1.602e-19**2 * n)
        kept_ixes = np.logical_and(diffs < 2, np.abs(val+160-df.xy.values) < 250)
        plt.plot(df.B.values[kept_ixes], df.xy.values[kept_ixes], 'b.')
        reses[n] = df.xy.values[kept_ixes]
    plt.xlabel('B (T)')
    plt.ylabel(r'$\rho_{xy}$ $(\Omega)$')

    ax = plt.subplot(gs[1])
    xs = []; ys = [];
    for res in reses:
        plt.plot([res for _ in reses[res]], 1./reses[res], 'o', c="#F8766D")
        xs.extend([res for _ in reses[res]])
        ys.extend(1./reses[res])
    p = np.polyfit(xs, ys, 1)
    plt.annotate(r'$1/\rho_{xy}=$' + '{:.2e}'.format(p[0]) + r'$ \nu$', xy=(.1,.8), xycoords='axes fraction')
    plt.plot(xs, p[0]*np.array(xs)+p[1], 'k')

    plt.xlabel(r'Label ($\nu$)')
    plt.ylabel(r'$1/\rho_{xy}$ ($\Omega^{-1}$)')

    plt.tight_layout()
    plt.savefig(fileout)
    plt.close(fig)

def quad_func(x, a, b, c, d, e, f, g, h,i, j, k):
    return j*x**3 + a*x**2 + b*x + c + (k*x**5 + i*x**4 + h*x**3 + d*x**2 + e*x)*np.sin(f/x + g)

def get_temp(info, fileout, ftemps, m_star):
    fig = plt.figure(figsize=(10,6))
    print 'plotting temp to', fileout

    handles = []
    big_x = []
    big_y = []
    big_b = []
    big_c = []
    colors = []
    sorted_keys = sorted(ftemps.keys(), key=lambda x: ftemps[x]['temp'][0])
    for f in sorted_keys:
        df,x_shift,p_shift,constant = info[f]

        # sweep up with error from sweep down
        df = df[np.logical_and(np.logical_and(df.sweep == 1, df.B < 1.7), df.B>.51)]

        #plt.plot(df.B.values, df.xx.values)

        #p0= [10, -1.27, 20, 1, 1, 32.2, 1, 1, 1, 1, 1]
        #p,_ = curve_fit(quad_func, df.B.values, df.xx.values, maxfev=3000, p0=p0)
        #plt.plot(df.B.values, [quad_func(val, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10]) for val in df.B.values], 'k--')
        j,a,b,c = 0.784398873668, 37.584194207, -39.3592341353, 35.7144229706
        if f[:5] == '10_19':
            correct_b = df.B.values+0.025
            imp_y = [(df.xx.values[ix] - (j*val**3 + a*val**2+b*val+c)) / df.xx.values[0] for ix,val in enumerate(df.B.values)]
        else:
            correct_b = df.B.values
            imp_y = [(df.xx.values[ix] - (j*val**3 + a*val**2+b*val+c)) / df.xx.values[0] for ix,val in enumerate(df.B.values)]
        p = plt.plot(correct_b, imp_y)

        label_name = str(ftemps[f]['temp'][0]) + r'$^{+' + str(ftemps[f]['temp'][2] - ftemps[f]['temp'][0]) + '}' + '_{-' + str(ftemps[f]['temp'][0] - ftemps[f]['temp'][1]) + '}$ K'
        handles.append(mpatches.Patch(color= p[0].get_color(), label=label_name))
        colors.append(p[0].get_color())
        #plt.annotate(str(p), xy=(.1,.8), xycoords='axes fraction')

        ex_ixes = argrelextrema(np.array(imp_y) , np.greater)[0]
        ex_ixes = np.append(ex_ixes, argrelextrema(np.array(imp_y) , np.less)[0])
        plt.plot( correct_b[ex_ixes], np.array(imp_y)[ex_ixes],'b.')

        #big_y.extend(np.log(np.abs(np.array(imp_y)[ex_ixes])))
        xi = 1.67*ftemps[f]['temp'][0] / correct_b[ex_ixes] #1.38e-23 * 2 * np.pi**2 * ftemps[f]['temp'][0] / (1.055e-34 * 1.602e-19 * correct_b[ex_ixes] / m_star)
        big_y.append(np.log(np.abs(np.array(imp_y)[ex_ixes])))
        big_x.append(np.log( xi / np.sinh(xi) ))

        big_b.append(correct_b[ex_ixes])
        big_c.append(np.log( np.abs(imp_y)[ex_ixes] * np.sinh(xi) / xi ))

    big_y = np.array(big_y)
    big_x = np.array(big_x)
    big_b = np.array(big_b)
    big_c = np.array(big_c)

    plt.legend(handles=handles, fontsize=15)
    plt.xlabel(r'B (T)')
    plt.ylabel(r'$\Delta \rho_{xx}/\rho_0$')

    plt.tight_layout()
    plt.savefig(fileout + '_0.pdf')
    plt.close(fig)

    fig = plt.figure(figsize=(10,6))

    ps = []
    for col_ix in range(big_y.shape[1]):
        plt.plot(big_x[:,col_ix], big_y[:,col_ix], '.')
        p = np.polyfit(big_x[:,col_ix], big_y[:,col_ix], 1)
        ps.append(p[1])
    #print m_star, np.mean(ps), np.std(ps), max(ps), min(ps)

    plt.tight_layout()
    plt.savefig(fileout + '_1.pdf')
    plt.close(fig)

    # Dingle plot
    fig = plt.figure(figsize=(10,6))

    for row_ix in range(len(big_b)):
        xi = 1.67*ftemps[sorted_keys[row_ix]]['temp'][0] / big_b[row_ix]
        plt.plot(1./big_b[row_ix], big_y[row_ix], 'o', c=colors[row_ix])
    p = np.polyfit(1./big_b.flatten(), big_y.flatten(), 1)
    plt.plot(1./big_b.flatten(), 1./big_b.flatten() * p[0] + p[1], 'k')
    plt.legend(handles=handles, fontsize=15)
    plt.xlabel(r'$1/B$ $(T^{-1})$')
    plt.ylabel(r'$\log(|\rho_{xx}/\rho_0|)$')

    plt.annotate(r'$y = ' + str(round(p[0],2)) + 'x' + ' + ' + str(round(p[1],2)) + r'$', xy=(.6,.8), xycoords='axes fraction')

    # extremes = (min(np.log(xi/ np.sinh(xi))), max(np.log(xi/ np.sinh(xi))))
    # plt.plot(extremes, extremes, 'k')

    plt.tight_layout()
    plt.savefig(fileout + '_2.pdf')
    plt.close(fig)

    # Experiment plot
    fig = plt.figure(figsize=(10,6))

    ps = []
    for row_ix in range(len(big_b)):
        xi = 1.67*ftemps[sorted_keys[row_ix]]['temp'][0] / big_b[row_ix]
        plt.plot(np.log(xi/ np.sinh(xi)), big_y[row_ix], 'o', c=colors[row_ix])
        p = np.polyfit(np.log(xi/ np.sinh(xi)), big_y[row_ix], 1)
        plt.plot(np.log(xi/ np.sinh(xi)), p[0]*np.log(xi/ np.sinh(xi)) + p[1], c=colors[row_ix])
        ps.append(p[0])
    print np.mean(ps)
    plt.legend(handles=handles, fontsize=15)
    plt.xlabel(r'$\log(\xi / \sinh(\xi))$')
    plt.ylabel(r'$\log(|\rho_{xx}/\rho_0|)$')

    plt.annotate(r'$m^* = 0.113m_e$', xy = (.4,.8), xycoords = 'axes fraction')

    # extremes = (min(np.log(xi/ np.sinh(xi))), max(np.log(xi/ np.sinh(xi))))
    # plt.plot(extremes, extremes, 'k')

    plt.tight_layout()
    plt.savefig(fileout + '_3.pdf')
    plt.close(fig)

    return np.mean(ps)


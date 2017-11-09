import numpy as np
import csv
import os
import math
import matplotlib.pyplot as plt
from scipy.stats import linregress
import matplotlib as mpl
mpl.style.use('ggplot')
mpl.rcParams['font.size'] = 20

infile = '10_19_2017_Run3_Jonah_Katie.txt'
path = '../data/'
infile = path + infile

with open(infile,'r') as fh:
    reader = csv.reader(fh, delimiter='\t')
    data = []
    for row in reader:
        if len(row)>0 and row[0][0].isalpha():
            continue
        else:
            data.append(np.array(row))
data = np.stack(data).astype(float)

tmax = np.amax(data[:,3])
MaxBcut = 25
tmincut = 10
cuts = []
rho_xy = []
for i in range(0,990):
    Bcut = MaxBcut - i/1000.*MaxBcut
    data_new = data[data[:,0]<Bcut]
    data_new = data_new[data_new[:,3]>tmincut]
    data_new = data_new[data_new[:,3]<tmax/2]
    B = .1192* data_new[:,0]
    Y = data_new[:,2]/5/45.5*10 #(should have another times 10^4)
    Y2 = data_new[:,1]*.35/45.5*(10**9)*10**(-4)
    try:
        slope, intercept, r_value, p_value, std_err = linregress(B,Y)
    except:
        continue
    print('fit' + 'y=' + str(slope)+'x+' + str(intercept))
    cuts.append(Bcut)
    rho_xy.append(slope*10000) #(10^4 times actual slope, accounts for Y value)
cuts = np.asarray(cuts) * .1192
plt.plot(cuts, rho_xy)
plt.xlabel('Max Magnetic Field (T)')
plt.ylabel(r'$\frac{d\rho_{xy}}{dB}e$')
#plt.ylabel('$d\rho/dB$ * 1000')
plt.show()

Bcut = 1000
cuts = []
rho_xy = []
tmincutmax = 700
for i in range(0, 990):
    tmincut = i/1000.*tmincutmax
    data_new = data[data[:,0]<Bcut]
    data_new = data_new[data_new[:,3]>tmincut]
    data_new = data_new[data_new[:,3]<tmax/2]
    B = .1192* data_new[:,0]
    Y = data_new[:,2]/5/45.5*10 #(should have another times 10^4)
    Y2 = data_new[:,1]*.35/45.5*(10**9)*10**(-4)
    try:
        slope, intercept, r_value, p_value, std_err = linregress(B,Y)
    except:
        continue
    print('fit' + 'y=' + str(slope)+'x+' + str(intercept))
    cuts.append(tmincut)
    rho_xy.append(slope*10000) #(10^4 times actual slope, accounts for Y value)

cuts = np.asarray(cuts)
plt.plot(cuts, rho_xy)
plt.xlabel('Min Time (s)')
plt.ylabel(r'$\frac{d\rho_{xy}}{dB}e$')
#plt.ylabel('$d\rho/dB$ * 1000')
plt.show()

x = np.random.random_sample(1000)
x = x
y = slope*x + intercept
plt.plot(B,Y2)
#plt.show()
plt.plot(B,Y, label='Classical Hall Effect')
plt.plot(x,y,color='green',label='Linear Fit')
#plt.show()

#hbar=1.05*10**(-34)
#e = 1.6*10**(-19)
#Y = data[:,2]/45.5*(10**9)*10**(-3)/5
#B = .1192* data[:,0]
#plt.plot(B,Y)
#x = x*5.5
#for i in range(2,10):
#    Ynew = np.full(1000, 2*math.pi*hbar/(e**2)/i + 131.132864502)
#    plt.plot(x, Ynew, color = 'green')
#plt.show()

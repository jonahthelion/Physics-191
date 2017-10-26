import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from scipy.stats import linregress
import matplotlib as mpl

infile = '10_19_2017_Run1_Jonah_Katie.txt'
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
MaxBcut = 10
tmincut = 30
cuts = []
rho_xy = []
rho_xx = []
for i in range(100, 1001):
    Bcut = i/1000.*MaxBcut
    print("TEST2: " + str(Bcut))
    data_new = data[data[:,0]<Bcut]
    data_new = data_new[data_new[:,3]>tmincut]
    data_new = data_new[data_new[:,3]>tmax/2]
    B = .1192* data_new[:,0]
    Y = data_new[:,2]/45.5*(10**9)
    Y2 = data_new[:,1]*.35/45.5*(10**9)
    slope, intercept, r_value, p_value, std_err = linregress(B,Y)
    slope2, intercept2, r_value2, p_value2, std_err2 = linregress(B,Y2)
    cuts.append(Bcut)
    rho_xy.append(slope)
    rho_xx.append(intercept2)
for i in range(len(cuts)):
    print(cuts[i])
    print(rho_xy[i])
    print(rho_xx[i])
plt.plot(cuts, rho_xx)
plt.show()
#print('fit' + 'y=' + str(slope)+'x+' + str(intercept))
x = np.random.random_sample(1000)
x = x/2*.1192
#y = slope*x + intercept
plt.plot(B,Y2)
#plt.plot(x,y,color='green')
plt.show()

hbar=1.05*10**(-34)
e = 1.6*10**(-19)
Y = data[:,2]/45.5*(10**9)
plt.plot(B,Y)
plt.plot(x, 2*Pi*hbar/(e**2)*x)

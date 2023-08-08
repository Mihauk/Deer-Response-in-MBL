import pickle
import numpy as np
#from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def func(x,a,b):
	return (a*np.log(x))+b
'''
def short_dis(x, y, m, k):
		d = np.abs(y-m*x-k)/np.sqrt(1+m**2)
		return np.argmin(d)
'''

with open('/home/abhishek/Documents/ny project/data/new/hv10_N10_eav50_dis100_t1012_d_rel_tsi_spin_echo_s11_d9_rc_up_Ngrid200.pickle', 'rb') as data:
	temp=pickle.load(data)
'''
with open('/home/abhishek/Documents/ny project/data/hv20_N8_eav50_dis1000_t112_d_rel_tsi_spin_echo_s13_s26_Ngrid200_spa_corr.pickle', 'rb') as data:
	temp1=pickle.load(data)

with open('/home/abhishek/Documents/ny project/data/hv20_N8_eav50_dis1000_t112_d_rel_tsi_spin_echo_s11_s28_Ngrid200_spa_corr.pickle', 'rb') as data:
	temp2=pickle.load(data)
'''

dt = temp[0]
D = temp[1]
sp = temp[2]
'''
D1 = temp1[1]
sp1 = temp1[2]
D2 = temp2[1]
sp2 = temp2[2]
#s = np.arange(3)
'''
s=[2,3,4,5]
#s=[2,3,4,5,6,7,8,9]

De = D.mean(axis=0)
Deer = De.mean(axis=0)

spi = sp.mean(axis=0)
spin = spi.mean(axis=0)
'''
De1 = D1.mean(axis=0)
Deer1 = De1.mean(axis=0)

spi1 = sp1.mean(axis=0)
spin1 = spi1.mean(axis=0)

De2 = D2.mean(axis=0)
Deer2 = De2.mean(axis=0)

spi2 = sp2.mean(axis=0)
spin2 = spi2.mean(axis=0)
'''
#er = stats.sem(De, axis=0)
er = np.std(De, axis=0)
ers = np.std(spi, axis=0)
'''
er1 = np.std(De1, axis=0)
ers1 = np.std(spi1, axis=0)

er2 = np.std(De2, axis=0)
ers2 = np.std(spi2, axis=0)
'''
'''
sat_Deer = np.zeros((4))
std_sat_Deer = np.zeros((4))

x = (np.where(dt>=100)[0])[0]
sat_Deer[0] = (Deer[0,x::]).mean()
sat_Deer[1] = (Deer[1,x::]).mean()
sat_Deer[2] = (Deer[2,x::]).mean()
sat_Deer[3] = (Deer[3,x::]).mean()

std_sat_Deer[0] = np.std((Deer[0,x::]))
std_sat_Deer[1] = np.std((Deer[1,x::]))
std_sat_Deer[2] = np.std((Deer[2,x::]))
std_sat_Deer[3] = np.std((Deer[3,x::]))
'''

'''
#plt.errorbar(dt, spin, yerr=ers, label=r"$spin-echo$")
#plt.errorbar(dt, Deer[0], yerr=er[0], label=r"$r=1$")
plt.errorbar(dt, Deer[1], yerr=er[1], label=r"$r=2$")
plt.errorbar(dt, Deer[2], yerr=er[2], label=r"$r=3$")
plt.errorbar(dt, Deer[3], yerr=er[3], label=r"$r=4$")
plt.errorbar(dt, Deer[4], yerr=er[4], label=r"$r=5$")
plt.errorbar(dt, Deer[5], yerr=er[5], label=r"$r=6$")
plt.errorbar(dt, Deer[6], yerr=er[6], label=r"$r=7$")
plt.errorbar(dt, Deer[7], yerr=er[7], label=r"$r=8$")
plt.errorbar(dt, Deer[8], yerr=er[8], label=r"$r=9$")
'''
'''
plt.errorbar(dt, spin2, yerr=ers2, label=r"$spin-echo,d=6$")
plt.errorbar(dt, Deer2, yerr=er2, label=r"$DEER,d=6$")
'''

'''
i = np.where(np.logical_and(dt>=5,dt<=850))[0]
popt, pcov = curve_fit(func, dt[i], Deer[1,i])
plt.plot(dt, func(dt, *popt), '-', label='fit, r=2, slope = %f' %(popt[0]))

i1 = np.where(np.logical_and(dt>=45,dt<=11500))[0]
popt1, pcov1 = curve_fit(func, dt[i1], Deer[2,i1])
plt.plot(dt, func(dt, *popt1), '-', label='fit, r=3, slope = %f' %(popt1[0]))


i2 = np.where(np.logical_and(dt>=250,dt<=60000))[0]
popt2, pcov2 = curve_fit(func, dt[i2], Deer[3,i2])
plt.plot(dt, func(dt, *popt2), '-', label='fit, r=4, slope = %f' %(popt2[0]))


i3 = np.where(np.logical_and(dt>=350,dt<=10000000))[0]
popt3, pcov3 = curve_fit(func, dt[i3], Deer[4,i3])
plt.plot(dt, func(dt, *popt3), '-', label='fit, r=5, slope = %f' %(popt3[0]))

i4 = np.where(np.logical_and(dt>=5000,dt<=200000000))[0]
popt4, pcov4 = curve_fit(func, dt[i4], Deer[5,i4])
plt.plot(dt, func(dt, *popt4), '-', label='fit, r=6, slope = %f' %(popt4[0]))

i5 = np.where(np.logical_and(dt>=50000,dt<=4000000000))[0]
popt5, pcov5 = curve_fit(func, dt[i5], Deer[6,i5])
plt.plot(dt, func(dt, *popt5), '-', label='fit, r=7, slope = %f' %(popt5[0]))

i6 = np.where(np.logical_and(dt>=250000,dt<=20000000000))[0]
popt6, pcov6 = curve_fit(func, dt[i6], Deer[7,i6])
plt.plot(dt, func(dt, *popt6), '-', label='fit, r=8, slope = %f' %(popt6[0]))

i7 = np.where(np.logical_and(dt>=500000,dt<=200000000000))[0]
popt7, pcov7 = curve_fit(func, dt[i7], Deer[8,i7])
plt.plot(dt, func(dt, *popt7), '-', label='fit, r=9, slope = %f' %(popt7[0]))
'''

'''
slp = np.zeros((8))
#intrcpt = np.zeros((3))
slp[0]=popt[0]
slp[1]=popt1[0]
slp[2]=popt2[0]
slp[3]=popt3[0]
slp[4]=popt4[0]
slp[5]=popt5[0]
slp[6]=popt6[0]
slp[7]=popt7[0]

plt.plot(s,slp,'o--')
'''

intrcpt = np.zeros((4))
intrcpt[0]=541.363#678.616 #5.02781 #bulk #pbc#541.363 #8.692 #obc#376.914 #8.33858
intrcpt[1]=15995#14960.2 #20.7969 #bulk #pbc15995 #34.691 #obc#18007.9 #21.9389
intrcpt[2]=171585#729864 #127.42 #bulk #pbc171585 #101.303 #obc#587916 #337.7
intrcpt[3]=497866#15273700 #265.612 #bulk #pbc497866 #234.224 #obc#12416800 #523.132
'''
intrcpt[4]=#279224000 #1302.76
intrcpt[5]=#7337880000 #13055.2
intrcpt[6]=#43986500000 #12071.8
intrcpt[7]=#599587000000 #70171.4
'''
plt.plot(s,intrcpt,'o--')



#plt.errorbar(s, sat_Deer, yerr=std_sat_Deer, label=r"$hv=5$")
#plt.xticks([0,1,2,3])
#plt.yticks([0,0.2,0.4,0.5])
#plt.xlabel(r"$t$", fontsize=18)
plt.xlabel(r"$r$", fontsize=18)
#plt.ylabel(r"$Slope$", fontsize=18)
plt.ylabel(r"$Sat-time$", fontsize=18)
#plt.ylabel(r"$\langle\langle D(t) \rangle\rangle$", fontsize=18)
#plt.ylabel(r"$SD$", fontsize=18)
#plt.ylabel(r"$C_{zz}$", fontsize=18)
plt.yscale('log')
#plt.xscale('log')
#plt.legend()
plt.show()
import pickle
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

with open('/home/abhishek/Documents/ny project/data/hv5_N8_eav50_dis100_t100000_s1to4_rc_Ngrid200_spin_echo.pickle', 'rb') as data:
	temp=pickle.load(data)
with open('/home/abhishek/Documents/ny project/data/hv10_N8_eav50_dis100_t100000_s1to4_rc_Ngrid200_spin_echo.pickle', 'rb') as data:
	temp1=pickle.load(data)

dt = temp[0]
D = temp[1]
dt1 = temp1[0]
D1 = temp1[1]

s = np.arange(4)

De = D.mean(axis=0)
Deer = De.mean(axis=0)
De1 = D1.mean(axis=0)
Deer1 = De1.mean(axis=0)

#e = stats.sem(D, axis=0)
er = np.std(De, axis=0)
er1 = np.std(De1, axis=0)

sat_Deer = np.zeros((4))
std_sat_Deer = np.zeros((4))
sat_Deer1 = np.zeros((4))
std_sat_Deer1 = np.zeros((4))

x = (np.where(dt>=100)[0])[0]
sat_Deer[0] = (Deer[0,x::]).mean()
sat_Deer[1] = (Deer[1,x::]).mean()
sat_Deer[2] = (Deer[2,x::]).mean()
sat_Deer[3] = (Deer[3,x::]).mean()

std_sat_Deer[0] = np.std((Deer[0,x::]))
std_sat_Deer[1] = np.std((Deer[1,x::]))
std_sat_Deer[2] = np.std((Deer[2,x::]))
std_sat_Deer[3] = np.std((Deer[3,x::]))


y = (np.where(dt1>=100)[0])[0]
sat_Deer1[0] = (Deer1[0,y::]).mean()
sat_Deer1[1] = (Deer1[1,y::]).mean()
sat_Deer1[2] = (Deer1[2,y::]).mean()
sat_Deer1[3] = (Deer1[3,y::]).mean()

std_sat_Deer1[0] = np.std((Deer1[0,x::]))
std_sat_Deer1[1] = np.std((Deer1[1,x::]))
std_sat_Deer1[2] = np.std((Deer1[2,x::]))
std_sat_Deer1[3] = np.std((Deer1[3,x::]))


#X,Y = np.meshgrid(dt, d)
#fig, ax = plt.subplots()

#p = ax.pcolor(X, Y, Deer)
#cb = fig.colorbar(p)

#plt.plot(d, Deer[:,199], label=r"$t=150$")

#plt.errorbar(dt, Deer, yerr=er, label=r"$hv=5$")
#plt.errorbar(dt, Deer[1], yerr=er[1], label=r"$d=1$")
#plt.errorbar(dt, Deer[2], yerr=er[2], label=r"$d=2$")
#plt.errorbar(dt, Deer[3], yerr=er[3], label=r"$d=3$")

'''
plt.errorbar(dt, Deer[0], yerr=er[0], label=r"$S=0, hv=5$")
plt.errorbar(dt, Deer[1], yerr=er[1], label=r"$S=1, hv=5$")
plt.errorbar(dt, Deer[2], yerr=er[2], label=r"$S=2, hv=5$")
plt.errorbar(dt, Deer[3], yerr=er[3], label=r"$S=3, hv=5$")

plt.errorbar(dt1, Deer1[0], yerr=er[0], label=r"$S=0, hv=10$")
plt.errorbar(dt1, Deer1[1], yerr=er[1], label=r"$S=1, hv=10$")
plt.errorbar(dt1, Deer1[2], yerr=er[2], label=r"$S=2, hv=10$")
plt.errorbar(dt1, Deer1[3], yerr=er[3], label=r"$S=3, hv=10$")
'''

plt.errorbar(s, sat_Deer, yerr=std_sat_Deer, label=r"$hv=5$")
plt.errorbar(s, sat_Deer1, yerr=std_sat_Deer1, label=r"$hv=10$")
plt.xticks([0,1,2,3])
#plt.yticks([0.2,0.4,0.5])
#y=D.mean(axis=1)
#plt.plot(dt, D[22,3,3])
#plt.xlabel(r"$t$", fontsize=18)
plt.xlabel(r"$s$", fontsize=18)
#plt.ylabel(r"$d$", fontsize=18)
plt.ylabel(r"$\langle\langle D(t) \rangle\rangle$", fontsize=18)
#plt.xscale('log')
plt.legend()
plt.show()
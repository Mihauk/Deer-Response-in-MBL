import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('/home/abhishek/Documents/ny project/data/deer_hv5_N10_tav.pickle', 'rb') as data:
	temp=pickle.load(data)

dt = temp[0]
D = temp[1]

De = D.mean(axis=0)
Deer = De.mean(axis=0)

plt.plot(dt, Deer)
#plt.xticks([50,100,150,200], " ")
plt.yticks([0.5,0.4,0.3,0.2,0.1,0])
plt.xlabel(r"$t$", fontsize=18)
plt.ylabel(r"$\langle\langle D(t) \rangle\rangle$", fontsize=18)
#plt.legend()
plt.show()

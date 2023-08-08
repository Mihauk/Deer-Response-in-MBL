'''
Program: Code for Deer response thermally averaged over 50 eigenstates for single disorder realization.
Name: Abhishek Raj
'''
import numpy as np
import scipy as sc
import cm
from numpy import linalg as la
import matplotlib.pyplot as plt

# Function defined for pi/2 Pulse operator on site i
def pulse(i):
	P = 1/np.sqrt(2)
	if i == 1:
		P = np.dot(P,(I - 2j*cm.spin("Y", c, N)))
	elif i == 2:
		P = P**l
		for ii in range(l):
			P = np.dot(P,(I - 2j*cm.spin("Y", (ii+1+c+d)%N, N)))
	return (P)

N = 10 #length of spin chain
c = 2 #index of spin site 1 (starting from 0)
d = 3 #no of spins between the spin site 1 and 2
l = 2 #no. of spins in the spin site 2
J = 1 #nearest neighbour coupling strength
#J_p = 0.1*J #uncomment if using random field xxz model.
t_max =  150
Ngrid = 300
samples = 50
dt = np.linspace(0, t_max, Ngrid)

I = cm.spin("I", 0, N) #identity operator in 2**N hilbert space

P_pio2_1 = pulse(1) #pi/2 pulse on spin site 1.
P_pio2_2 = pulse(2) #pi/2 pulse on spin site 2.
P_pi_1 = np.dot(P_pio2_1, P_pio2_1) #pi pulse on spin site 1.
P_pi_2 = np.dot(P_pio2_2, P_pio2_2) #pi pulse on spin site 2.

sigma_z_1 = cm.spin("Z", c, N)
D = np.zeros((samples, Ngrid))

h_v = (10*np.random.rand(N)-5) #disorder
H = cm.h_rfh_obc( J, N, h_v)
#H = cm.h_rfxxz( J, J_prime, N, h_v) #random field xxz model with open boundary condition
e, v = la.eigh(H)
v_dagg = np.conj(v.T)

for j in range(samples):
	print(j)
	s = 1
	r = np.random.randint(2, size=N)
	r[c] = 1
	for ii in range(N):
		if r[ii] == 1:
			s = np.kron(s,cm.up)
		else:
			s = np.kron(s,cm.down)
	psi_0 = s
	phi_0 = np.dot(P_pio2_1,psi_0)
	for i in range(Ngrid):
		#phi_t = np.dot(P_pi_1, cm.psi_at_t( e, v, v_dagg, dt[i]/2, phi_0))
		phi_t = np.dot(P_pi_1,np.dot(P_pi_2,cm.psi_at_t( e, v, v_dagg, dt[i]/2, phi_0)))
		chi_t = np.dot(P_pio2_1,cm.psi_at_t( e, v, v_dagg, dt[i]/2, phi_t))
		chi_t_dagg = np.conj(chi_t.T)
		D[j, i] = cm.check_real(np.dot(chi_t_dagg, np.dot(sigma_z_1,chi_t)))
		#D[j, i] = np.absolute(temp)

Deer = D.mean(axis=0)

plt.plot(dt, Deer)
#plt.xticks([50,100,150,200], " ")
#plt.yticks([0.6,0.4,0.2,0,-0.2])
plt.xlabel(r"$t$", fontsize=18)
plt.ylabel(r"$\langle D(t) \rangle$", fontsize=18)
#plt.legend()
plt.show()

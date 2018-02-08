from IsingEnsemble import IsingEnsemble
import numpy as np
from numpy import gradient
import matplotlib.pylab as plt
from matplotlib import rc
from tqdm import tqdm # progress bar

rc('text', usetex=True)

model=IsingEnsemble(2,5,3)


dT = 0.05
T = np.arange(0.1,20,dT)
betas = 1/T

stat={}

stat["beta"]=betas
keys = model.keys

for key in keys:
	stat[key]=np.array([])

print "\n Calculating statistics for different temperatures..."
for beta in tqdm(betas):
	model.set_(beta=beta)
	for key in keys:
		stat[key]=np.append(stat[key],model.stats[key])


plt.plot(T,stat["energy"])
plt.ylabel(key.capitalize())
plt.xlabel("Temperature")
plt.show()


from ising import IsingDist
import numpy as np
from scipy.ndimage import convolve
import matplotlib.pylab as plt 
from tqdm import tqdm
from saveobject import save_obj
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

dT=0.05
T = np.arange(dT,6,dT)
betas = 1/T
mus = np.logspace(-4.0,0,num=10)
size=4**4
samples = 1024

fig, axs = plt.subplots(3,2,figsize=(6.3,4.2))

values = {}
values["betas"] = betas
values["mu"] = mus

for beta in tqdm(betas):
	for mu in mus:
		dist = IsingDist(size,beta,mu)
		dist.sample(samples)
		dist.calcEV()
		values[(beta,mu)] = [dist.energy, dist.magnet]
		del dist

save_obj(values, "values")







from ising import IsingDist
import numpy as np
from scipy.ndimage import convolve
import matplotlib.pylab as plt 
from tqdm import tqdm
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


beta = 1
mu = 0
size=4**0
samples = 1024
dist = IsingDist(size,beta,mu)
dist.sample(samples)
dist.calcEV()

fig, axs = plt.subplots(3,2,figsize=(6.3,4.2))
n=[2**0,2**2,2**4,2**6,2**8,2**10]

x=dist.configs

for j in tqdm(range(6)):
	m = n[j]
	j+=1
	ax = plt.subplot(2, 3, j)
	ax.imshow(x[m],interpolation=None,cmap="pink")
	ax.set_title(r"$t=%i$"%m)
	ax.set_xticks([])
	ax.set_yticks([])

titlestr = r'$N = %i$'%size
fig.text(0.01, 0.5, titlestr ,fontsize=12,rotation=90)
plt.tight_layout()
plt.savefig('image'+"%i"%size+".pdf", bbox_inches='tight')
plt.show()




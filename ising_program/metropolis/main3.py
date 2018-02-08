from ising import IsingDist
import numpy as np
from scipy.ndimage import convolve
import matplotlib.pylab as plt 
from tqdm import tqdm
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


beta = 1
mu = 0
size=4**4
samples = 1024


fig, axs = plt.subplots()
n=[2**0,2**2,2**4,2**6,2**8,2**10]

for beta in [0.1,0.2,0.5,0.7,1.0]:
	print beta
	dist = IsingDist(size,beta,mu)
	dist.sample(samples)
	dist.calcEV()
	plt.plot(beta,dist.magnet[1])
	del dist

plt.xlabel(r"Time $t$")
plt.ylabel(r"Average Magnetization $\bar M$")
plt.legend()
plt.tight_layout()
plt.savefig('magvstime.pdf', bbox_inches='tight')
plt.show()




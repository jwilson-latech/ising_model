from ising import IsingDist
import numpy as np
from scipy.ndimage import convolve
import matplotlib.pylab as plt 
from tqdm import tqdm
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


beta = 1
mu = 0
size=42**4
samples = 1024


fig, axs = plt.subplots()
N=[2**0,2**1,2**2,2**3,2**4,2**5]

for n in N:
	print n
	dist = IsingDist(n**2,beta,mu)
	dist.sample(samples)
	dist.calcEV()
	plt.plot(n**2,dist.magnet[1])
	del dist

plt.xlabel(r"System Size $N^2$")
plt.ylabel(r"Average Magnetization $\bar M$")
plt.legend()
plt.tight_layout()
plt.savefig('magvssize.pdf', bbox_inches='tight')
plt.show()


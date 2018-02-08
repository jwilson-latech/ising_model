from saveobject import load_obj
import matplotlib.pylab as plt 
import numpy as np
from numpy import sqrt
from scipy.ndimage import convolve

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

values = load_obj("values")
mus =  np.array(values["mu"])
betas = np.array(values["betas"])

def smooth(y, box_pts=20):
    ker = np.ones(box_pts)/box_pts
    y_smooth = convolve(y,ker,mode='reflect')
    return y_smooth

E =  np.array([[values[(beta,mu)][0][0] for beta in betas] for mu in mus])
dE = np.array([[values[(beta,mu)][0][1] for beta in betas] for mu in mus])
M =  np.array([[values[(beta,mu)][1][0] for beta in betas] for mu in mus])
dM = np.array([[values[(beta,mu)][1][1] for beta in betas] for mu in mus])

loc = 0
# for i in range(10):
# 	i=9-i
#  	y = smooth(M[i])
#  	dy = smooth(dM[i])
#  	plt.plot(1/betas,y,label=r"$\mu=%04.4f$"%mus[i])
#  	plt.fill_between(1/betas,y-dy,y+dy,alpha=0.25)
#  	plt.xlabel(r'Temperature $1/\tilde\beta$')
#  	plt.ylabel(r'Average Magnetization $\bar M$')
#  	plt.legend()
# plt.savefig('magnet.pdf')
# plt.show()


for i in range(10):
	print mus[i]
	y = smooth(E[i])
	dy = smooth(dE[i])
	plt.plot(1/betas,y)
	plt.fill_between(1/betas,y-dy,y+dy,alpha=0.25,label=r"$\mu=%04.4f$"%mus[i])
	plt.legend()
	plt.xlabel(r'Temperature $1/\tilde\beta$')
  	plt.ylabel(r'Average Energy $\bar E$')
plt.savefig('energy.pdf')
plt.show()






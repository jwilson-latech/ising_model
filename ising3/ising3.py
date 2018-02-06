import numpy as np
from numpy import log
from scipy.ndimage import convolve
from tqdm import tqdm

def safeexp(x):
	if x==0:
		return 1
	else:
		return x**x

class Ising():

	def __init__(self,size,beta,mu,steps=1,random=True):
		self.size = size
		self.beta = beta
		self.mu = mu
		a = 1.0
		b = 0.0
		self.window = np.array([[b,a,b],\
								[a,0,a],\
								[b,a,b]])
		if random == True:
			self.config = 1+2*np.random.randint(0,2,size=(size,size))-1
		if random == False:
			self.config = 1+0*np.random.randint(0,2,size=(size,size))-1
		self.flipcount = 0 
		self.steps = steps

	def __del__(self):
		return None

	def __enter__(self):
		return None
	def __exit__(self):
		return None

	def update(self):
		self.mcmove()
		return None

	def mcmove(self):
		size = self.size
		config = self.config
		altconfig = 2*np.random.randint(0,2,size=(size,size))-1
		therand = np.random.rand(size,size)
		dH = -self.beta*(self.config*2\
			*convolve(config,self.window,mode='wrap')+2*self.mu*config)
		configacc = np.where(log(therand)<dH,altconfig,config)
		self.flipcount+=np.sum(abs(config-configacc))/2
		self.config = configacc
		return None

	def evolve(self,steps):
		for i in range(steps):
			self.update()

	def thermofuncs(self,key):
		if key == "energy":
			return self.calcEnergy()
		if key == "magnetization":
			return self.calcMag()
		if key == "population":
			return self.calcPop()
		if key == "config_population":
			return self.calcConfigPop()
		if key == "entropy":
		 	return self.calcEntropy()
		if key == "config_entropy":
		 	return self.calcConfigEntropy()
		if key == "flips":
		 	return self.calcNumFlips()

	def calcEnergy(self):
		config = self.config
		dE = -self.config*(2\
			*convolve(config,self.window,mode='wrap')+2*self.mu)
		return np.sum(dE)/(8.0*self.size**2)

	def calcMag(self):
		config = self.config
		return abs(np.sum(config)/(1.0*self.size**2))

	def calcPop(self):
		config = self.config
		size = self.size
		popup = np.count_nonzero(config == 1)/(1.0*size**2)
		pop = max(popup,1-popup)
		return pop

	def calcEntropy(self):
		p1 = self.calcPop()
		p2 = 1-p1
		entropy = -log(safeexp(p1)*safeexp(p2))
		return entropy/log(2)

	def calcConfigPop(self):
		config = self.config
		size = self.size
		filconfig = config*convolve(config,self.window,mode='wrap')/2
		c0 = np.count_nonzero(filconfig == -2)/(1.0*size**2)
		c1 = np.count_nonzero(filconfig == -1)/(1.0*size**2)
		c2 = np.count_nonzero(filconfig == +0)/(1.0*size**2)
		c3 = np.count_nonzero(filconfig == +1)/(1.0*size**2)
		c4 = np.count_nonzero(filconfig == +2)/(1.0*size**2)
		return np.array([c0,c1,c2,c3,c4])

	def calcConfigEntropy(self):
		config = self.config
		size = self.size
		[c0,c1,c2,c3,c4] = self.calcConfigPop()
		return -log(safeexp(c0)*safeexp(c1)*safeexp(c2)*safeexp(c3)*safeexp(c4))

	def calcNumFlips(self):
		return self.flipcount/(1.0*self.steps*self.size**2)











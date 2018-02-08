import numpy as np
from numpy import log
from scipy.ndimage import convolve
from tqdm import tqdm

def safeexp(x):
	if x==0:
		return 1
	else:
		return x**x

class make_distribution():

	def __init__(self,size,beta,mu,steps=1,random=True):
		self.size = size
		self.beta = beta
		self.mu = mu
		a = 1.0
		b = 0.0

		self.window = np.array([[b,a,b],\
								[a,0,a],\
								[b,a,b]])

		self.configs = []
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

	def evolve(self,steps):
		for i in range(steps):
			self.update()

	def update(self):
		self.mcmove()
		self.configs.append(self.configs)
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

	def calcEnergy(self):
		config = self.config
		dE = -self.config*(2\
			*convolve(config,self.window,mode='wrap')+2*self.mu)
		return np.sum(dE)/(8.0*self.size**2)












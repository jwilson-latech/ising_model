import numpy as np
from numpy import exp, log
from scipy.ndimage import convolve
from tqdm import tqdm
import time
from saveobject import save_obj

class IsingDist():

	def __init__(self,size,beta,mu):
		self.name = str(int(time.time()))
		self.size = size
		self.beta = beta
		self.mu = mu
		a = 1.0
		b = 0.0

		self.window = np.array([[b,a,b],\
								[a,0,a],\
								[b,a,b]])

		self.config = 2*np.random.randint(0,2,size=(size,size))-1
		self.configs = [self.config]

	def __del__(self):
		return None

	def __enter__(self):
		return None
	def __exit__(self):
		return None

	def sample(self,steps):
		for i in range(steps):
			self.update()
		self.configs = np.array(self.configs)

	def update(self):
		self.mcmove()
		self.configs.append(self.config)

	def mcmove(self):
		size = self.size
		config = self.config
		altconfig = 2*np.random.randint(0,2,size=(size,size))-1
		therand = np.random.rand(size,size)
		dH = -self.beta*(self.config*2\
			*convolve(config,self.window,mode='wrap')+2*self.mu*config)
		configacc = np.where(log(therand)<dH,altconfig,config)
		self.config = configacc
		return None

	def calcEnergy(self):
		config = self.config
		dE = -self.config*(2\
			*convolve(config,self.window,mode='wrap')+2*self.mu)
		return np.sum(dE)

	def calcDist(self):
		self.energy_()
		self.magnet_()
		self.prob_()

	def calcEV(self):
		self.calcDist()
		E = self.energy_dist
		M = self.magnet_dist
		P = self.prob_dist

		eE = np.dot(E,P)
		sE = np.sqrt(np.dot(E**2,P)-eE**2)
		self.energy = [eE, sE]

		eM = np.dot(M,P)
		sM = np.sqrt(np.dot(M**2,P)-eM**2)
		self.magnet = [eM, sM]



	def energy_(self):
		size = self.size
		configs = self.configs
		window = np.array([self.window])
		sum_ej = convolve(configs,window,mode='wrap')
		dE = -configs*(sum_ej+self.mu)
		self.energy_dist = np.sum(dE,axis=(1,2))/(1.0*size**2)

	def magnet_(self):
		size = self.size
		configs = self.configs
		M = abs(np.sum(configs,axis=(1,2)))
		self.magnet_dist = M/(1.0*size**2)

	def prob_(self):
		beta = self.beta
		energy = self.energy_dist
		estar = np.min(energy)
		dE = energy - estar
		logP = -beta*dE - log(np.sum(exp(-beta*dE)))
		self.prob_dist = exp(logP)

	def pickle(self):
		save_obj(self, self.name)









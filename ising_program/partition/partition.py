import numpy as np
from numpy import exp, sqrt, log 
from scipy.ndimage import convolve
import matplotlib.pylab as plt
import matplotlib.animation as animation
import itertools
from tqdm import tqdm

def safeEntropy(p):
	if p ==0:
		return 0
	else:
		return -p*log(p)

def safeEntropy2(p):
	if p ==0:
		return 0
	else:
		return p*(log(p))**2

vSE = np.vectorize(safeEntropy)
vSE2 = np.vectorize(safeEntropy2)

class IsingEnsemble():
	def __init__(self,size,beta=1.0,mu=0.0,**kwargs):
		self.size = size
		self.beta = beta
		self.mu = mu
		self.kwargs = kwargs
		self.stats = {}
		a = 1.0
		b = 0.0
		self.window = np.array([[b,a,b],\
								[a,0,a],\
								[b,a,b]])

		self.makeConfigs()
		self.default()

	def default(self):
		kwargs = self.kwargs
		if "onlyconfigs" in kwargs:
			if kwargs["onlyconfigs"]==True:
				return None
		else: 
			self.calcEnergy()
			self.sortByEnergy()
			self.calcProb()
			self.calcStats()

	def set_(self,**kwargs):
		if "mu" in kwargs:
			self.mu = kwargs["mu"]
			self.calcEnergy()
			self.sortByEnergy()
		if "beta" in kwargs:
			self.beta = kwargs["beta"]

		self.calcPart()
		self.calcProb()
		self.calcStats()

	def makeConfigs(self):
		configs=[]
		m = self.size
		n = m**2
		for i in tqdm(range(1<<n)):
			s=bin(i)[2:]
			s='0'*(n-len(s))+s
			p = np.array(map(int,list(s)))
			p=2*p.reshape((m,m))-1
			configs.append(p)
		self.configs = np.array(configs)

	def calcEnergy(self):
		configs = self.configs
		size = self.size
		window = self.window
		mu = self.mu 
		energies = []
		for config in configs:
			energy = -np.sum(config*convolve(config,window,mode='wrap')-config*mu)
			energies.append(energy)
		self.energies = np.array(energies)

	def sortByEnergy(self):
		sortby = np.array(self.energies)
		configs = self.configs
		energies = self.energies
		idxs = sortby.argsort()
		self.energies = energies[idxs]
		self.configs = configs[idxs]

	def calcProb(self):
		beta = self.beta
		energies = self.energies
		totalEnergy = np.total(self.energy)
		log_probabilities = -energies*beta/totalEnergy
		probabilities =  exp(log_probabilities)
		self.probabilities = probabilities

	def calcStats(self):
		energies = self.energies/self.size**2
		prob = self.probabilities
		energy = np.sum(energies*prob)
		energy2 = np.sum(energies**2*prob)
		sigma = sqrt(energy2-energy**2)
		self.stats["energy"] = energy
		self.stats["energystd"] = sigma
		
		entropy = np.sum(vSE(self.probabilities))
		entropy2 = np.sum(vSE2(self.probabilities))
		sigma = entropy2-entropy**2
		self.stats["entropy"] = entropy
		self.stats["entropystd"] = sigma











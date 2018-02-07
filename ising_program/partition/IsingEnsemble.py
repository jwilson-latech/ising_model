import numpy as np
from numpy import exp, sqrt, log 
from scipy.ndimage import convolve
import matplotlib.pylab as plt
import matplotlib.animation as animation
import itertools
from tqdm import tqdm
import time
from saveobject import save_obj

def eLog(p):
	if p ==0:
		return 0
	else:
		return log(p)

veLog = np.vectorize(eLog)

class IsingEnsemble():
	def __init__(self,size,beta=1.0,mu=0.0,**kwargs):
		self.keys = ["energy","entropy","magnetization"]
		self.name = str(int(time.time()))
		self.size = size
		self.beta = beta
		self.mu = mu
		self.kwargs = kwargs
		self.stats = {}
		self.values = {}
		self.stats["mu"] = mu
		self.stats["beta"] = beta
		a = 1.0
		b = 0.0
		self.window = np.array([[[b,a,b],\
								 [a,0,a],\
								 [b,a,b]]])

		self.makeConfigs()
		self.default()

	def default(self):
		kwargs = self.kwargs
		if "onlyconfigs" in kwargs:
			if kwargs["onlyconfigs"]==True:
				return None
		else: 
			self.calcEnergy()
			self.calcProb()
			self.calcMag()
			self.calcStats()

	def set_(self,**kwargs):
		if "mu" in kwargs:
			self.mu = kwargs["mu"]
			self.stats["mu"] = self.mu
			self.calcEnergy()
		if "beta" in kwargs:
			self.beta = kwargs["beta"]
			self.stats["T"] = self.beta

		self.calcProb()
		self.calcMag()
		self.calcStats()

	def makeConfigs(self):
		""" 
		Creates all possible configurations of an size*size matrix 
		with entries of 1 or -1.

		"""
		print "\n Calculating all possible configurations..."
		configs=[]
		m = self.size
		n = m**2
		for i in tqdm(range(1<<n)):
			s = bin(i)[2:]
			s = '0'*(n-len(s))+s
			p = np.array(map(int,list(s)))
			p = 2*p.reshape((m,m))-1
			configs.append(p)
		self.configs = np.array(configs)

	def calcEnergy(self):
		configs = self.configs
		size = self.size
		window = self.window
		mu = self.mu 
		energies = -np.sum(configs*(convolve(configs,window,mode='wrap')+mu),axis=(1,2))
		self.values["energy"] = np.array(energies)

	def sortByEnergy(self):
		energies = self.values["energy"]
		sortby = np.array(energies)
		configs = self.configs
		energies = energies
		idxs = sortby.argsort()
		self.values["energies"] = energies[idxs]
		self.configs = configs[idxs]

	def calcProb_old(self):
		partition = self.partition
		themax = np.amax(partition)
		newpart = partition/themax
		total = np.sum(newpart)
		self.probabilities = newpart/total

	def calcProb(self):
		beta = self.beta
		energies = self.values["energy"]
		E_ = np.min(energies)
		dE = (energies - E_)*beta
		log_sum = log(np.sum(exp(-dE)))
		self.values["probabilities"] = exp(-dE-log_sum)

	def calcMag(self):
		configs = self.configs
		mag = np.sum(configs, axis=(1,2))
		self.values["magnetization"] = np.abs(mag)

	def calcStats(self):
		prob = self.values["probabilities"]
		keys = self.keys
		for key in keys:
			if key == "entropy":
				self.stats[key] = -np.sum(prob*veLog(prob))
				self.stats[key+"std"] = None
			else:
				values = self.values[key]/self.size**2
				expvalue = np.sum(values*prob)
				expvalue2 = np.sum(values**2*prob)
				std = sqrt(expvalue2-expvalue**2)
				self.stats[key] = expvalue
				self.stats[key+"std"] = std

		self.pickle()

	def pickle(self):
		save_obj(self, self.name)

class Ising():
	def __init__(self,size,beta_range=[1.0],mu_range=[0.0],**kwargs):
		self.keys = ["energy","entropy","magnetization"]
		self.size = size
		self.beta_range = beta_range
		self.mu_range = mu_range
		self.kwargs = kwargs

	def iterator(self):
		return None














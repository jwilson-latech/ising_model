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

class Interface(object):

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











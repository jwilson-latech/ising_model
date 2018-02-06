import numpy as np
from ising3 import Ising
from tqdm import tqdm

keys = ["energy","magnetization","population","entropy","flips","config_entropy","config_population"]

class Ensemble():
	def __init__(self,size,betaRange,muRange,steps,times,random=True):
		global keys
		self.size = size
		self.betaRange = betaRange
		self.muRange = muRange
		self.steps = steps
		self.times = times
		self.keys = keys
		self.betaN = len(betaRange)
		self.muN = len(muRange)
		self.makeDicts()
		self.stats={}
		for key in keys:
			self.stats[key] = []
			self.stats[key+"std"] = []

		self.random=random
		if random == False:
			print "Proceeding with nonrandom initial conditions."

	def makeDicts(self):
		self.beta = {}
		self.mu = {}
		for i in range(self.betaN):
			self.beta[i]=self.betaRange[i]
		for i in range(self.muN):
			self.mu[i]=self.muRange[i]

	def getStats(self):
		self.stats["mu"]=self.muRange
		self.stats["beta"]=self.betaRange
		for i in tqdm(range(self.betaN)):
			stats1={}
			for j in range(self.muN):
				size = self.size
				beta = self.beta[i]
				mu = self.mu[j]
				steps = self.steps
				times = self.times
				self.calcStats(size,beta,mu,steps,times,)


	def calcStats(self,size,beta,mu,steps,times):
		global keys
		arr = {}
		for key in keys:
			arr[key]=[]
		for i in range(times):
			model = Ising(size,beta,mu,steps,self.random)
			model.evolve(steps)
			for key in keys:
				prop=model.thermofuncs(key)
				arr[key].append(prop)
			del model
		for key in keys:
			self.stats[key].append(np.mean(arr[key],0))
			self.stats[key+"std"].append(np.std(arr[key]))
		

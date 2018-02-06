# coding=utf-8
from ising3 import Ising
from ensemble import Ensemble
import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as animation
from saveobject import save_obj

N = 100
steps = 1000
repeat=10
res = 0.01
b1=0
b2=10

B=np.arange(b1,b2,res)
B=B[B!=0]
B=1/B

M=np.array([0])

ensemble = Ensemble(N,B,M,steps,repeat,False)
ensemble.getStats()

beta = ensemble.beta
mu = ensemble.mu

stats=ensemble.stats

save_obj(stats,"stats7")







# keys = ["energy","magnetization","population","entropy"]
# def calcStats(size,beta,mu,steps,times):
# 	global keys
# 	stats = {}
# 	arr = {}
# 	for key in keys:
# 		arr[key]=[]
# 	for i in range(times):
# 		model = ising(size,beta,mu)
# 		model.evolve(steps)
# 		for key in keys:
# 			prop=model.thermofuncs(key)
# 			arr[key].append(prop)
# 		del model
# 	for key in keys:
# 		stats[key] = [np.mean(arr[key]),np.std(arr[key])]
# 	return stats

# stats = calcStats(100,1,0,1000,10)
# print "##############################"
# for key in keys:
# 	val = stats[key][0]
# 	std = stats[key][1]
# 	print key+": ",val,"±",std
# print "##############################"



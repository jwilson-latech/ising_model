import numpy as np
from scipy.ndimage import convolve
import matplotlib.pylab as plt
import matplotlib.animation as animation

class Ising():

	def __init__(self,size,beta):
		self.size = size
		self.beta = beta
		self.window = np.array([[0,1,0],\
								[1,0,1],\
								[0,1,0]])
		self.config = 2*np.random.randint(0,2,size=(size,size))-1

	def update(self):
		self.mcmove()
		return None

	def mcmove(self):
		size = self.size
		config = self.config
		altconfig = 2*np.random.randint(0,2,size=(size,size))-1
		dH = -self.config*2*convolve(config,self.window,mode='wrap')
		config = np.where(dH==8,-config,config)
		config = np.where(dH==4,-config,config)
		config = np.where(dH==0,altconfig,config)
		config = np.where(dH==-4,config,config)
		config = np.where(dH==-8,config,config)
		self.config = config
		return None

model = Ising(100,1)
for i in range(1000):
	model.update()










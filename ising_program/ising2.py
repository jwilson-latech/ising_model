import numpy as np
from numpy import log,exp,tanh
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from numpy import genfromtxt


mu=-100
db = 2*genfromtxt('latech.csv', delimiter='\t')-1

# Set up plotter
fig = plt.figure()
ims=[]

class Ising():

	def __init__(self,size,beta,mu,steps):
		self.size = size
		self.beta = beta
		self.mu = mu
		self.window = np.array([[0,1,0],\
								[1,0,1],\
								[0,1,0]])
		self.config = 2*np.random.randint(0,2,size=(size,size))-1
		self.m = -steps+100
		self.steps = steps

	def update(self):
		self.mcmove()
		return None

	def mcmove(self):
		self.m = self.m + 1
		m = self.m
		size = self.size
		config = self.config
		altconfig = 2*np.random.randint(0,2,size=(size,size))-1
		therand = np.random.rand(size,size)
		dH = -self.beta*self.config*(
			2*convolve(config,self.window,mode='wrap') + 0*(1+tanh(m))*db)
		config = np.where(log(therand)<dH,altconfig,config)
		self.config = config
		return None

	def evolve(self):
		global ims
		for i in range(self.steps):
			m1 = model.config
			model.update()
			m2 = model.config
			m = m2
			im=plt.imshow(m,vmin=np.min(m))
			ims.append([im])




# Running Simulation
model = Ising(450,1,0,1000)
model.evolve()



ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)

ani.save('ising.gif', dpi=80, writer='imagemagick')
plt.show()






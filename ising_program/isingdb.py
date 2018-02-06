import numpy as np
from numpy import log,exp,tanh
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter,fourier_gaussian

from numpy import genfromtxt


mu= 0.5
db = genfromtxt('hilbert.csv', delimiter='\t')
db = 2*db-1

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
		self.m = 0
		self.steps = steps

	def update(self):
		self.mcmove()
		return None

	def mcmove(self):
		self.m = self.m + 1
		m = self.m
		size = self.size
		steps=self.steps
		config = self.config
		altconfig = 2*np.random.randint(0,2,size=(size,size))-1
		therand = np.random.rand(size,size)

		dH = -self.beta*self.config*(
			2*convolve(config,self.window,mode='wrap') 
			- mu*db)
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
model = Ising(260,2,0,300)
model.evolve()



ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)

ani.save('reddit.gif', dpi=80, writer='imagemagick')
plt.show()






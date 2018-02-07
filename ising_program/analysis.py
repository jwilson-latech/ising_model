import numpy as np
from numpy import log, sqrt,exp,gradient
from saveobject import load_obj
import matplotlib 
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pylab as plt
from scipy.ndimage import convolve

Stats5=load_obj("stats6")
Stats6=load_obj("stats7")

def stats(key):
    return Stats5[key][2:-2]

def stats01(key):
    return Stats6[key][2:-2]

def smooth(y, box_pts=20):
    ker = np.ones(box_pts)/box_pts
    y_smooth = convolve(y,ker,mode='reflect')
    return y_smooth

b=np.array(stats("beta"))
b2=np.array(stats01("beta"))
kT=1/b
kT2=1/b2


xkey = "config_population"
ykey = "energy"

#E = np.array(stats("energy"))
#S = np.array(stats("entropy"))
#S1 = np.array(stats("config_entropy"))
M = np.array(stats("magnetization"))
err=np.array(stats("magnetizationstd"))
M2 = np.array(stats01("magnetization"))
err2 = np.array(stats01("magnetizationstd"))

fix, ax = plt.subplots()

#ax.plot(kT, smooth(2*S*log(2)) ,'k-',label=r"$S$")

#ax.errorbar(kT,smooth(M,40),yerr=err/sqrt(10),label=r"$M01$")
ax.plot(kT, 0.5*(M+M2),"k.",label=r"$M01$")
ax.plot(kT2, M,"r.",label=r"$C_V$")
#ax.errorbar(kT,M,yerr=err/sqrt(10),label=r"$M01$")
#ax.errorbar(kT2, M2 ,yerr=err2/10,label=r"$C_V$")

#ax.plot(kT,smooth(M,40) ,'r-',label=r"$M$")
#ax.legend()
#ax.plot(kT, smooth(x1),label=r"$1$")
#ax.plot(kT, smooth(x2),label=r"$2$")
#ax.plot(kT, smooth(x3),label=r"$3$")
#ax.plot(kT, smooth(x4),label=r"$4$")
#ax.plot(kT, energy1/8,label=r"$4$")
#ax.plot(kT, energy,label=r"$4$")



#ax.plot(1/b, smooth(val,10),'k-')

#ax.errorbar(1/b, val,yerr=err,fmt='.',label="Config Entropy",capsize=2)


#p = smooth(np.gradient(smooth(val_y,20)),20)
#p=p/sum(p)
#print np.dot(p,1/b)

#ax.plot(1/b,val_x,"r.")
#ax.errorbar(1/b, val_x, yerr=err_x, fmt='.')

#plt.loglog(val_x,val_y,"r.")
#std_x=np.array(stats(key+"std"))

#np.sum(val * 1/b)

#slen=20
#err1=smooth(val+std,slen)
#err2=smooth(val-std,slen)
#val_s=smooth(val,slen)
#plt.plot(1/b,val,'.',color="red")
#plt.plot(1/b,smooth(val,slen),'-',color="black")

#ax = axs[0]
#ax.errorbar(xvals, avg_cpu, yerr=std_cpu, fmt='bs')


#Cv=np.gradient(val_s)
#Cvs=smooth(Cv,slen)

#plt.plot(1/b,val_s,'-',color="black")
#plt.plot(1/b,val_s,'-',color="black")
#plt.plot(1/b,Cv,'.',color="red")
#plt.plot(1/b,Cv,'-',color="black")
#Cvn = Cv/np.sum(Cv)


#print np.dot(Cvn,1/b)
#plt.fill_between(1/b, err1, err2)
#plt.axvline(x=8/log(2),color="red")


plt.show()

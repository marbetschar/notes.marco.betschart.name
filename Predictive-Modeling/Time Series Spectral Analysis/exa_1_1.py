import numpy as np
import matplotlib.pyplot as plt





# initializing the period
T = 2*np.pi

# initializing the time interval
timespan = np.linspace(-2*T, 4*T, 1000)


# initializing the angular frequency
omega = 2*np.pi/T

# initializing the values of the amplitudes ak and bk
nb_coeff = 150 # try: nb_coeff = 2, 3, 4, ...
k = np.arange(1,nb_coeff+1,1)

a0 = 0
ak = 0*k
bk = 2*( 1 - (-1)**k )/(np.pi*k)


# computing the signal
s = [a0/2 + sum( ak*np.cos(k*omega*t) + bk*np.sin(k*omega*t) ) for t in timespan]


# plotting results
plt.plot(timespan, s, 'b')
plt.xlabel('time')
plt.ylabel('Value of $s$')
plt.show()
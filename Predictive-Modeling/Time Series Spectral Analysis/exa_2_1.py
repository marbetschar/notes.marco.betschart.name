import numpy as np
import matplotlib.pyplot as plt





#%%
# example for 1st function
import numpy as np
import matplotlib.pyplot as plt
# function: f(t) = t*(1 - t)
# interval: [0, 1[

#% degree of trigonometric polynomial and number of sample points
N = 7 # try N = 2, 3, ...
n = 2*N + 1

# determining time interval, value of function and Fourier coefficients
T = 1
omega = 2*np.pi/T
t = T*np.arange(0,n)/n
x = t*(1-t)

# calculating Fourier coefficients
c = np.fft.fft(x)/n



# calculating the approximated x's (x_app <-> x approximated)
tt = np.linspace(0, T, 1000)
x_app = np.real( c[0]*np.exp(1j*0*tt) )
f = tt*(1 - tt)

for k in np.arange(1,N+1):
    x_app = x_app + 2*np.real(c[k]*np.exp(1j*k*omega*tt))



# plotting results
plt.plot(tt, x_app, 'b-', label = 'trigonometric polynomial')
plt.plot(t, x, 'ro', label = 'interpolation points')
plt.plot(tt, f, 'g', label = 'function $f$')
plt.legend()
plt.show()




#%%
# example for 2nd function

# function: f(t) = 2 + 3cos(t) + 4sin(t) + cos(5t)
# interval: [0, 2pi[

# degree of trigonometric polynomial and number of sample points
N = 5 # try N = 2, 3, ...
n = 2*N + 1

# determining time interval, value of function and Fourier coefficients
T = 2*np.pi
omega = 2*np.pi/T
t = T*np.arange(0,n)/n
x = 2 + 3*np.cos(t) + 4*np.sin(t) + 1*np.cos(5*t)

# calculating Fourier coefficients
c = np.fft.fft(x)/n



# calculating the approximated x's (x_app <-> x approximated)
tt = np.linspace(0, T, 1000)
x_app = np.real( c[0]*np.exp(1j*0*tt) )
f = 2 + 3*np.cos(tt) + 4*np.sin(tt) + 1*np.cos(5*tt)

for k in np.arange(1,N+1):
    x_app = x_app + 2*np.real(c[k]*np.exp(1j*k*omega*tt))



# plotting results
plt.plot(tt, x_app, 'b-', label = 'trigonometric polynomial')
plt.plot(t, x, 'ro', label = 'interpolation points')
plt.plot(tt, f, 'g', label = 'function $f$')
plt.legend()
plt.show()
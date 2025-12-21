import numpy as np
import matplotlib.pyplot as plt





# example with
# function:
# f(t) = -1 for t in [-pi, 0[
# f(t) = 1 for t in [0, pi[
# interval: [-pi, pi[

# initializing the values of the coefficients
nb_coeff = 50

n = np.arange(1,nb_coeff+1,1)

a0 = 0
an = 0*n
bn = 2*( 1 - (-1)**n )/(np.pi*n)

A = np.sqrt(an**2 + bn**2)
A = np.insert(A,0,a0)


# plotting the amplitudes
plt.stem(np.arange(0,nb_coeff+1), A)
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.show()
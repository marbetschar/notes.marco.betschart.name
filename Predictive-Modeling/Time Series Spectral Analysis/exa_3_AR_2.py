import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess





# Example AR(2): Autoregressive of order 2
# X_j - phi_1*X_{j-1} - phi_2*X_{j-2} = W_j


# here: W_j ~ N(0,1)
# number of time steps:
n = 500
# value of parameters
phi_1 = 1
phi_2 = -0.9


# one realisation
model = ArmaProcess(ar = np.array([1, -phi_1, -phi_2]), ma = np.array([1]))
# testing for stationary
model.isstationary
x = model.generate_sample(nsample = n)
plt.plot(x)
plt.title('Autoregressive (AR(2))')
plt.show()

c = np.fft.fft(x)/n
Per = np.abs(c)**2


Freq = np.arange(1,n-1)/n
plt.stem(Freq[0:math.floor(n/2)+1], Per[0:math.floor(n/2)+1])
plt.xlabel('Frequency')
plt.ylabel('$|c_n|^2$')
plt.title('Periodogram of AR(2)')
plt.show()




# other realizations
x = model.generate_sample(nsample = n)

# computing the Fourier coefficients
c = np.fft.fft(x)/n

Per = np.abs(c)**2
Freq = np.arange(1,n-1)/n
plt.stem(Freq[0:math.floor(n/2)+1], Per[0:math.floor(n/2)+1])
plt.xlabel('Frequency')
plt.ylabel('$|c_n|^2$')
plt.show()




# number of simulations
nb_sim = 1000

# initializing list to collect data
df_X = []

for ind_sim in range(nb_sim):
    x = model.generate_sample(nsample = n)
    
    # computing the Fourier coefficients
    c = np.fft.fft(x)/n
    
    Per = np.abs(c)**2
    
    df_X.append(Per[0:math.floor(n/2)+1])


df_X = pd.DataFrame(df_X)

Per_avg = df_X.mean(axis=0)


# theoretical spectral density
s = np.linspace(0, 0.5, 1000)
f = 1 / ( 1 - 2*phi_1*(1 - phi_2)*np.cos(2*np.pi*s) - 2*phi_2*np.cos(4*np.pi*s) + phi_1**2 + phi_2**2 )


plt.stem(Freq[0:math.floor(n/2)+1], Per_avg[0:math.floor(n/2)+1]*n, 'k')
plt.plot(s, f, color = 'r')
plt.xlabel('Frequency')
plt.ylabel('$n|c_n|^2$')
plt.title('Periodogram of AR(2)')
plt.show()

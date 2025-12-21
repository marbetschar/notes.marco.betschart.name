import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess





# Example MA(1): Moving Average of order 1
# X_j = W_j + \theta_1 W_{j-1}


# here: W_j ~ N(0,1)
# number of time steps
n = 500
# value of parameters
theta_1 = 0.5


# one realisation
model = ArmaProcess(ar = np.array([1]), ma = np.array([1, theta_1]))
# testing for invertibility
model.isinvertible
x = model.generate_sample(nsample = n)
plt.plot(x)
plt.title('Moving Average (MA(1))')
plt.show()

c = np.fft.fft(x)/n
Per = np.abs(c)**2


Freq = np.arange(1,n-1)/n
plt.stem(Freq[0:math.floor(n/2)+1], Per[0:math.floor(n/2)+1])
plt.xlabel('Frequency')
plt.ylabel('$|c_n|^2$')
plt.title('Periodogram of MA(1)')
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
f = 1 + 2*theta_1*np.cos(2*np.pi*s) + theta_1**2


plt.stem(Freq[0:math.floor(n/2)+1], Per_avg[0:math.floor(n/2)+1]*n, 'k')
plt.plot(s, f, color = 'r')
plt.xlabel('Frequency')
plt.ylabel('$n|c_n|^2$')
plt.title('Periodogram of MA(1)')
plt.show()


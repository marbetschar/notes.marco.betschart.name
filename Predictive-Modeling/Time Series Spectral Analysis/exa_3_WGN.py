import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd





# Example W.G.N.: White Gaussian Noise
# W_j <-> white noise


# here: W_j ~ N(0,1)
# number of time steps
n = 500

# one realisation
w = np.random.normal(size = n)
x = w
plt.plot(x)
plt.title('Gaussian White Noise')


c = np.fft.fft(x)/n
Per = np.abs(c)**2


Freq = np.arange(1,n-1)/n
plt.stem(Freq[0:math.floor(n/2)+1], Per[0:math.floor(n/2)+1])
plt.xlabel('Frequency')
plt.ylabel('$|c_n|^2$')
plt.title('Periodogram of W.G.N.')
plt.show()




# other realizations
x = np.random.normal(size = n)

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
    x = np.random.normal(size = n)
    
    # computing the Fourier coefficients
    c = np.fft.fft(x)/n
    
    Per = np.abs(c)**2
    
    df_X.append(Per[0:math.floor(n/2)+1])


df_X = pd.DataFrame(df_X)

Per_avg = df_X.mean(axis=0)


# theoretical spectral density
s = [0,0.5]
f = [1,1]


plt.stem(Freq[0:math.floor(n/2)+1], Per_avg[0:math.floor(n/2)+1]*n, 'k')
plt.plot(s, f, color = 'r')
plt.xlabel('Frequency')
plt.ylabel('$n|c_n|^2$')
plt.title('Periodogram of W.G.N')
plt.show()


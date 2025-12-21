import numpy as np
import matplotlib.pyplot as plt





#%%
# example "cleaning" noisy signal

# generating noisy signal
n = 500
t = np.linspace(0, 2*np.pi, n)

s = 0.75 + 1*np.cos(1*t) + 0.3*np.sin(1*t) + 0.5*np.cos(3*t) + 0.1*np.sin(6*t) + 0.4*np.random.normal(size = n)


# plotting signal
plt.plot(t,s)
plt.xlabel('Time $t$')
plt.ylabel('Signal $s$')
plt.show()

# computing the fft
c = np.fft.fft(s)/n


# plotting the 20 first amplitudes
plt.stem(np.arange(0, 21), abs(c[0:21]))
plt.xlabel('n')
plt.ylabel('|c_n|')


# filtering (c_TRA <-> c transformed)
threshold = 0.1
c_TRA = c.copy()
c_TRA[abs(c_TRA) < threshold] = 0
s_clean = np.real( n*np.fft.ifft(c_TRA) )


plt.plot(t, s, color = 'k', label = 'noisy signal')
plt.plot(t, s_clean, color = 'g', linewidth = 4, label = 'filtered signal')
plt.xlabel('Time $t$')
plt.ylabel('Signal $s$')
plt.legend()
plt.show()




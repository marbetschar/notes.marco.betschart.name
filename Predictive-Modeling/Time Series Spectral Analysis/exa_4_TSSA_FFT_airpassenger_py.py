




# Example TSSA with airpassenger FFT

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path_data = '../data/'


### # ### start: loading data

df_AP = pd.read_csv(path_data + 'AirPassengers.csv')

# creating pandas DateTimeIndex
dtindex = pd.DatetimeIndex(data = pd.to_datetime(df_AP['TravelDate']), freq = 'infer')

# setting as Index
df_AP.set_index(dtindex, inplace = True)
df_AP.drop( 'TravelDate', axis = 1, inplace = True)

# plotting data
plt.plot(df_AP)
plt.xlabel('time')
plt.ylabel('Passengers in 1000')
plt.grid()
plt.show()
### # ### end: loading data





### # ### start: selecting and renaming data

# setting variable values
AP = df_AP['Passengers'] # AP <-> air passengers

### # ### end: selecting and renaming data





### # ### start: FFT

# 1st Approch --- with original data
# computing the fft on original data
n = len(AP)
c_org = np.fft.fft(AP)/n # org <-> original

# plotting the moduli of the coefficients
# without c_0 and c_n with n >= 1 (i.e. the "positively indexed" coefficients)
n_pos = int(np.floor((n-1)/2))

# plotting result
plt.stem(np.arange(1,n_pos+1), abs(c_org[1:n_pos+1]))
plt.xlabel('$n$')
plt.ylabel('$|c_n|$')
plt.title('Moduli of the Coefficients (without $|c_0|$)')
plt.show()


# modulus of amplitudes
mod_amp = abs(c_org)

# considering the moduli of the amplitudes from c_0 and c_n with n >= 1 (positively indexed)
mod_amp = mod_amp[:np.floor(len(c_org) / 2).astype(int)]

# sorting mod_amp in descending order and getting the indices
ind = np.argsort(mod_amp)[::-1]
sor_mod_amp = mod_amp[ind]

# index "0" and index of harmonics with the 6 largest amplitudes
ind_k = ind[:7]

# printing out results as a table
result_df = pd.DataFrame({
    '|c_k|': sor_mod_amp[:7],
    'Index': ind_k
})
print(result_df)



# cancelling all coefficients from the FFT that are smaller than threshold
c_TRA = c_org.copy()
threshold = 15
c_TRA[abs(c_TRA) < threshold] = 0

# iFFT on the processed coefficients and representing result with the original data
y = np.real(np.fft.ifft(c_TRA)*n)



# plotting results
plt.plot(AP, 'k', label = 'Data')
plt.plot(AP.index, y, 'g', label = 'Trig. Poly.')
plt.ylabel('Passengers in 1000')
plt.title('Air Passengers')
plt.legend(loc = 'best')
plt.show()




# 2nd Approch --- with log transformed data
# (with Box-Cox transform with lambda = 0, therefore just taking the log)
AP_TRA = np.log(AP) # TRA <-> transformed

# computing the fft on transformed data
n = len(AP_TRA)
c_log = np.fft.fft(AP_TRA)/n # log <-> log

# plotting the moduli of the coefficients
# without c_0 and c_n with n >= 1 (i.e. the "positively indexed" coefficients)
n_pos = int(np.floor((n-1)/2))

# plotting result
plt.stem(np.arange(1,n_pos+1), abs(c_log[1:n_pos+1]))
plt.xlabel('$n$')
plt.ylabel('$|c_n|$')
plt.title('Moduli of the Coefficients (without $|c_0|$)')
plt.show()


# modulus of amplitudes
mod_amp = abs(c_log)

# considering the moduli of the amplitudes from c_0 and c_n with n >= 1 (positively indexed)
mod_amp = mod_amp[:np.floor(len(c_log) / 2).astype(int)]

# sorting mod_amp in descending order and getting the indices
ind = np.argsort(mod_amp)[::-1]
sor_mod_amp = mod_amp[ind]

# index "0" and index of harmonics with the 6 largest amplitudes
ind_k = ind[:7]

# printing out results as a table
result_df = pd.DataFrame({
    '|c_k|': sor_mod_amp[:7],
    'Index': ind_k
})
print(result_df)



# cancelling all coefficients from the FFT that are smaller than threshold
c_TRA = c_log.copy()
threshold = 0.055
c_TRA[abs(c_TRA) < threshold] = 0

# iFFT on the processed coefficients and representing result with the original data
y = np.real(np.fft.ifft(c_TRA)*n)



# plotting results
plt.plot(AP_TRA, 'k', label = 'Data')
plt.plot(AP_TRA.index, y, 'g', label = 'Trig. Poly.')
plt.ylabel('log(Passengers)')
plt.title('Air Passengers')
plt.legend(loc = 'best')
plt.show()




# 3rd Approch --- with differencing
AP_TRA_1 = AP_TRA.diff()

# computing the fft on transformed data with differencing once
n = len(AP_TRA_1.dropna())
c = np.fft.fft(AP_TRA_1.dropna())/n

# plotting the moduli of the coefficients
# without c_0 and c_n with n >= 1 (i.e. the "positively indexed" coefficients)
n_pos = int(np.floor((n-1)/2))

# plotting result
plt.stem(np.arange(1,n_pos+1), abs(c[1:n_pos+1]))
plt.xlabel('$n$')
plt.ylabel('$|c_n|$')
plt.title('Moduli of the Coefficients (without $|c_0|$)')
plt.show()


# modulus of amplitudes
mod_amp = abs(c)

# considering the moduli of the amplitudes from c_0 and c_n with n >= 1 (positively indexed)
mod_amp = mod_amp[:np.floor(len(c) / 2).astype(int)]

# sorting mod_amp in descending order and getting the indices
ind = np.argsort(mod_amp)[::-1]
sor_mod_amp = mod_amp[ind]

# index "0" and index of harmonics with the 6 largest amplitudes
ind_k = ind[:7]

# printing out results as a table
result_df = pd.DataFrame({
    '|c_k|': sor_mod_amp[:7],
    'Index': ind_k
})
print(result_df)



# cancelling all coefficients from the FFT that are smaller than threshold
c_TRA = c.copy()
threshold = 0.02
c_TRA[abs(c_TRA) < threshold] = 0
# reintroducing the value of coefficient c_0
c_TRA[0] = c[0]

# iFFT on the processed coefficients and representing result with the original data
y_TRA = np.real(np.fft.ifft(c_TRA)*n)

y = np.r_[AP_TRA[0], y_TRA].cumsum()



# plotting results
# transformed data
plt.plot(AP_TRA, 'k', label = 'Data')
plt.plot(AP_TRA.index, y, 'g', label = 'Trig. Poly.')
plt.ylabel('log(Passengers)')
plt.title('Air Passengers')
plt.legend(loc = 'best')
plt.show()

# original data
plt.plot(AP, 'k', label = 'Data')
plt.plot(AP.index, np.exp(y), 'g', label = 'Trig. Poly.')
plt.ylabel('Passengers in 1000')
plt.title('Air Passengers')
plt.legend(loc = 'best')
plt.show()
### # ### end: FFT





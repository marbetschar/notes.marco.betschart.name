




import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



path_data = "data/"

### # ### start: loading data

# 2022
df_data_ENG_CH_2022 = pd.read_csv(path_data + 'df_data_ENG_CH.csv', index_col = 'timestamp')

# inspecting the data
df_data_ENG_CH_2022.head()
df_data_ENG_CH_2022.tail()
df_data_ENG_CH_2022.shape
df_data_ENG_CH_2022.describe()
df_data_ENG_CH_2022.dtypes
df_data_ENG_CH_2022.info()

# preparing data (converting to Datetime Format)
dtindex = pd.DatetimeIndex(data = df_data_ENG_CH_2022.index, freq = 'infer')
df_data_ENG_CH_2022.set_index(dtindex, inplace = True)
del dtindex

# inspecting the data after preparation
df_data_ENG_CH_2022.head()
df_data_ENG_CH_2022.dtypes
df_data_ENG_CH_2022.info()



# 2023
df_data_ENG_CH_2023 = pd.read_csv(path_data + 'df_data_ENG_CH_2023.csv', index_col = 'timestamp')

# inspecting the data
df_data_ENG_CH_2023.head()
df_data_ENG_CH_2023.tail()
df_data_ENG_CH_2023.shape
df_data_ENG_CH_2023.describe()
df_data_ENG_CH_2023.dtypes
df_data_ENG_CH_2023.info()

# preparing data (converting to Datetime Format)
dtindex = pd.DatetimeIndex(data = df_data_ENG_CH_2023.index, freq = 'infer')
df_data_ENG_CH_2023.set_index(dtindex, inplace = True)
del dtindex

# inspecting the data after preparation
df_data_ENG_CH_2023.head()
df_data_ENG_CH_2023.dtypes
df_data_ENG_CH_2023.info()

### # ### end: loading data





### # ### start: selecting and renaming data

df_EC_train = df_data_ENG_CH_2022['2022-01-01':'2022-12-31']
df_EC_test = df_data_ENG_CH_2023['2023-01-01':'2023-12-31']


# dates for plot
s_date_plot = '2022-07-01' # s <-> start
e_date_plot = '2022-12-31' # e <-> end


# setting variable values
EC = df_EC_train['EndCons'] # EC <-> end consumption

### # ### end: selecting and renaming data





### # ### start: FFT

# FFT on the time series

# computing the fft
n = len(EC)
c = np.fft.fft(EC)/n

# plotting the modulus of the amplitudes
# without c_0 and c_n with n >= 1 (i.e. the "positively indexed" coefficients)
n_pos = int(np.floor((n-1)/2))

# plotting result
plt.stem(np.arange(1,n_pos+1), abs(c[1:n_pos+1]))
plt.xlabel('$n$')
plt.ylabel('$|c_n|$')
plt.title('Moduli of the Coefficients (without $|c_0|$)')



# filtering

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
threshold = 30000
c_TRA[abs(c_TRA) < threshold] = 0

# iFFT on the processed coefficients and representing result with the original data
y = np.real(np.fft.ifft(c_TRA)*n)

pred = y[0:len(df_EC_test.index)]

# plotting results
plt.plot(EC[s_date_plot:e_date_plot], 'k', label = 'train data')
plt.plot(df_EC_test['EndCons'], ':k', label = 'true data')
plt.plot(df_EC_test.index, pred, 'r', label = 'Prediction')
plt.ylabel('EC [kWh]')
plt.title('End Users Electrical Energy Consumption (Swiss controlblock)')
plt.legend(loc = 'best')


# determining diff and ratio time series
diff = df_EC_test['EndCons'] - pred
diff_ratio = diff / df_EC_test['EndCons']


# plotting
plt.subplots(2, 1)
plt.subplot(2, 1, 1)
plt.plot(diff)
plt.subplot(2, 1, 2)
plt.plot(diff_ratio)


# computing performance metrics
# Mean Absolute Error (MAE)
MAE = np.mean(abs(diff))

# Root Mean Square Error (RMSE)
RMSE = np.sqrt(np.mean(diff**2))

# Mean Absolute Percentage Error (MAPE)
MAPE = 100*np.mean(abs(diff_ratio))

# output of results
list(np.round(np.array([MAE, RMSE, MAPE]), 2))
# [139334.79, 168057.44, 9.56]

### # ### end: FFT





### # ### start: SARIMA

# applying "differencing" for lag 7
EC_TRA_7 = EC.diff(periods = 7)

# plotting
plt.plot(EC_TRA_7)
plt.xlabel('time')
plt.ylabel('EC [kWh]')
plt.title('Differencing: s = 7')
plt.grid()



# STEP: determining the values of "order" parameters P and Q (seasonal)

# plotting the empirical autocorrelation (acf) and empirical partial autocorrelation (pacf)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig = plt.figure(figsize = (14, 5))
ax1 = fig.add_subplot(1, 2, 1)
plot_acf(EC_TRA_7.dropna(), lags = 50, ax = ax1, title = 'Empirical Autocorrelation')
ax1.plot([7, 7], [-1, 1], ':r')
ax2 = fig.add_subplot(1, 2, 2)
plot_pacf(EC_TRA_7.dropna(), lags = 50, ax = ax2, title = 'Empirical Partial Autocorrelation')
ax2.plot([7, 7], [-1, 1], ':r')

# OK for P = 1 and Q = 1



# STEP: determine the values of "order" parameters p and q (non-seasonal)

from statsmodels.tsa.arima.model import ARIMA
from itertools import product
import seaborn as sns

p = np.arange(4)
q = np.arange(5)

AIC = np.zeros((len(p), len(q)))
for comb in product(p, q):
    p_ind = comb[0]
    q_ind = comb[1]
    model = ARIMA(EC, order = (p_ind, 0, q_ind), seasonal_order = (1, 1, 1, 7))
    res = model.fit()
    AIC[p_ind, q_ind] = res.aic


# plotting heatmap
plt.figure(figsize = (8, 6))
ax = sns.heatmap(AIC, cmap = 'coolwarm')
ax.set_xlabel('q')
ax.set_ylabel('p')
ax.set_title('Heatmap with different order parameters p and q ')
plt.show()

# OK for p = 3 and q = 1
result = ARIMA(EC, order = (3, 0, 1), seasonal_order = (1, 1, 1, 7)).fit()
result.aic




from statsmodels.tsa.arima.model import ARIMA

# fitting SARIMA model on data for 2022
model_sam = ARIMA(EC, order = (3, 0, 1), seasonal_order = (1, 1, 1, 7)).fit()

model_sam.summary()

# computing predictions
pred = model_sam.get_prediction(start = '2023-01-01', end = '2023-12-31')
pred = pred.prediction_results
pred = pred._forecasts[0]

plt.plot(EC[s_date_plot:e_date_plot], 'k', label = 'train data')
plt.plot(df_EC_test['EndCons'], ':k', label = 'true data')
plt.plot(df_EC_test.index, pred, 'r', label = 'Prediction')
plt.ylabel('EC [kWh]')
plt.title('End Users Electrical Energy Consumption (Swiss controlblock)')
plt.legend(loc = 'best')

### # ### end: SARIMA





### # ### start: SARIMAX

from statsmodels.tsa.statespace.sarimax import SARIMAX

# setting the period
T = len(EC)

# determing the angular frequency
omega = 2*np.pi / T

# initializing values for exogenous features
cons = np.repeat(np.mean(EC), len(EC))
cos1 = np.array(np.cos(np.arange(0,T)*omega))
sin1 = np.array(np.sin(np.arange(0,T)*omega))

val_exog = np.stack((cons, cos1, sin1)).T


model_samx = SARIMAX(EC,
                  exog = val_exog,
                  order = (3, 0, 1), seasonal_order = (1, 1, 1, 7)).fit()


model_samx.summary()

# computing predictions
pred = model_samx.predict(start = '2023-01-01', end = '2023-12-31', exog = val_exog)

plt.plot(EC[s_date_plot:e_date_plot], 'k', label = 'train data')
plt.plot(df_EC_test['EndCons'], ':k', label = 'true data')
plt.plot(df_EC_test.index, pred, 'r', label = 'Prediction')
plt.ylabel('EC [kWh]')
plt.title('End Users Electrical Energy Consumption (Swiss controlblock)')
plt.legend(loc = 'best')


# determining diff and ratio time series
diff = df_EC_test['EndCons'] - pred
diff_ratio = diff / df_EC_test['EndCons']


# plotting
plt.subplots(2, 1)
plt.subplot(2, 1, 1)
plt.plot(diff)
plt.subplot(2, 1, 2)
plt.plot(diff_ratio)


# computing performance metrics
# Mean Absolute Error (MAE)
MAE = np.mean(abs(diff))

# Root Mean Square Error (RMSE)
RMSE = np.sqrt(np.mean(diff**2))

# Mean Absolute Percentage Error (MAPE)
MAPE = 100*np.mean(abs(diff_ratio))

# output of results
list(np.round(np.array([MAE, RMSE, MAPE]), 2))
# [83294.1, 109602.9, 5.42]

### # ### end: SARIMAX





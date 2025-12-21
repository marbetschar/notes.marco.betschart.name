




# Example TSSA comparing predictions between SARIMA and FFT with airpassenger

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path_data = "../data/"

### # ### start: loading data

AirP = pd.read_csv(path_data + 'AirPassengers.csv')

# creating pandas DateTimeIndex
dtindex = pd.DatetimeIndex(data = pd.to_datetime(AirP['TravelDate']), freq = 'infer')

# setting as Index
AirP.set_index(dtindex, inplace = True)
AirP.drop('TravelDate', axis = 1, inplace = True)

# plotting data
plt.plot(AirP)
plt.xlabel('time')
plt.ylabel('Passengers in 1000')
plt.grid()

### # ### end: loading data





### # ### start: selecting and renaming data

# setting variable values
AP = AirP['Passengers'] # AP <-> air passengers

### # ### end: selecting and renaming data





### # ### start: SARIMA

# transforming the data
AP_TRA = np.log(AP) # TRA <-> transformed


# forecasting
from statsmodels.tsa.arima.model import ARIMA

# fitting SARIMA model on data from 1949-01 to 1958-12
model_opt = ARIMA(AP_TRA['1949':'1958'], order = (0, 1, 1), seasonal_order = (1, 1, 1, 12)).fit()

# computing predictions
pred = model_opt.get_prediction(start = '1959-01-01', end = '1960-12-01')
pred = pred.prediction_results
pred = pred._forecasts[0]

# plotting results in original variables (i.e. after inverse Box-Cox)
plt.plot(AP['1949':'1958'], '-k', label = 'train data')
plt.plot(AP['1959':'1960'], 'b', label = 'validation data')
plt.plot(AP['1959':'1960'].index, np.exp(pred), 'r', label = 'Prediction')
plt.ylabel('Passengers in 1000')
plt.legend()
plt.show()


# determining diff and ratio time series
diff = AP['1959':'1960'] - np.exp(pred)
diff_ratio = diff / AP['1959':'1960']


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
# [38.97, 42.75, 8.42]

### # ### end: SARIMA





### # ### start: FFT

# differencing once
AP_TRA_1 = AP_TRA.diff()

# computing the fft on transformed data with differencing once
n = len(AP_TRA_1.dropna())
c = np.fft.fft(AP_TRA_1.dropna())/n

# cancelling all coefficients from the FFT that are smaller than threshold
c_TRA = c.copy()
threshold = 0.02
c_TRA[abs(c_TRA) < threshold] = 0
# reintroducing the value of coefficient c_0
c_TRA[0] = c[0]

# iFFT on the processed coefficients and representing result with the original data
y_TRA = np.real(np.fft.ifft(c_TRA)*n)

y = np.r_[AP_TRA[0], y_TRA].cumsum()

# converting result into Series
y = pd.Series(y, index = AP_TRA.index)

pred = y['1959':'1960']

# plotting results in original variables (i.e. after inverse Box-Cox)
plt.plot(AP['1949':'1958'], '-k', label = 'train data')
plt.plot(AP['1959':'1960'], 'b', label = 'validation data')
plt.plot(np.exp(pred), 'r', label = 'Prediction')
plt.legend()
plt.show()


# determining diff and ratio time series
diff = AP['1959':'1960'] - np.exp(pred)
diff_ratio = diff / AP['1959':'1960']


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
# [27.39, 35.65, 5.73]

### # ### end: FFT

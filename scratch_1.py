import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller

print('oi')

pd.set_option('display.max_columns', 20)

register_matplotlib_converters()
# %matplotlib inline

dataset = pd.read_csv('AAPL.csv')
dataset['Mean'] = (dataset['Low'] + dataset['High']) / 2

steps = -1
dataset_for_prediction = dataset.copy()
dataset_for_prediction['Actual'] = dataset_for_prediction['Mean'].shift(steps)
dataset_for_prediction = dataset_for_prediction.dropna()
dataset_for_prediction['Date'] = pd.to_datetime(dataset_for_prediction['Date'])
dataset_for_prediction.index = dataset_for_prediction['Date']

sc_in = MinMaxScaler(feature_range=(0, 1))
scaled_input = sc_in.fit_transform(
    dataset_for_prediction[['Low', 'High', 'Open', 'Close', 'Volume', 'Adj Close', 'Mean']])
scaled_input = pd.DataFrame(scaled_input)
X = scaled_input

sc_out = MinMaxScaler(feature_range=(0, 1))
scaler_output = sc_out.fit_transform(dataset_for_prediction[['Actual']])
scaler_output = pd.DataFrame(scaler_output)
y = scaler_output

X.rename(columns={0: 'Low', 1: 'High', 2: 'Open', 3: 'Close', 4: 'Volume', 5: 'Adj Close', 6: 'Mean'}, inplace=True)
y.rename(columns={0: 'Stock Price next day'}, inplace=True)
y.index = dataset_for_prediction.index

# SPLITTING TO TRAINING / TEST
train_size = int(len(dataset) * 0.7)
test_size = int(len(dataset)) - train_size
train_X, train_y = X[:train_size].dropna(), y[:train_size].dropna()
test_X, test_y = X[train_size:].dropna(), y[train_size:].dropna()

#   DECOMPOSITION FOR TREND / SEASONALITY / NOISE
seas_d = sm.tsa.seasonal_decompose(X['Mean'], model='add', period=365)
fig = seas_d.plot()
fig.set_figheight(10)
fig.set_figwidth(10)


# Augmented Dickey-Fuller test
def test_adf(series, title=''):
    dfout = {}
    dftest = sm.tsa.adfuller(series.dropna(), autolag='AIC', regression='ct')
    for key, val in dftest[4].items():
        dfout[f'critical value ({key})'] = val
    if dftest[1] <= 0.05:
        print("Strong evidence against Null Hypothesis")
        print("Reject Null Hypothesis - Data is Stationary")
        print("Data is Stationary", title)
    else:
        print("Strong evidence for  Null Hypothesis")
        print("Accept Null Hypothesis - Data is not Stationary")
        print("Data is NOT Stationary for", title)


y_test = y['Stock Price next day'][:train_size].dropna()
test_adf(y_test, " Stock Price")

test_adf(y_test.diff(), 'Stock Price')

# ACF / PACF
fig, ax = plt.subplots(2, 1, figsize=(10, 5))
fig = sm.tsa.graphics.plot_acf(y_test, lags=50, ax=ax[0])
fig = sm.tsa.graphics.plot_pacf(y_test, lags=50, ax=ax[1])
plt.show()

print(y.head(5))
pass

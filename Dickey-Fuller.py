import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from statsmodels.tsa.stattools import adfuller
import pandas as pd


def generate_ar_process(lags, coefs, length):
    # cast coefs to np array
    coefs = np.array(coefs)

    # initial values
    series = [np.random.normal() for _ in range(lags)]

    for _ in range(length):
        # get previous values of the series, reversed
        prev_vals = series[-lags:][::-1]

        # get new value of time series
        new_val = np.sum(np.array(prev_vals) * coefs) + np.random.normal()

        series.append(new_val)
    return np.array(series)


def perform_adf_test(series):
    result = adfuller(series)
    print('ADF Statistic(zastat): %f' % result[0])
    pvalue = result[1]
    print(pvalue)
    print('cvdict: %f' % result[2])
    print('baselags: %f' % result[3])
    print(result[4])
    print(result[5])


# CUSTOM DATASET
# zastat, pval, cvdict, baselags, bpidx
df = pd.read_csv('inp.csv', header=None)
series_inp = np.array(df[2])
perform_adf_test(series_inp)



# AR(1) Process
# Stationary
ar_1_process = generate_ar_process(1, [.5], 100)
plt.figure(figsize=(10, 4))
plt.plot(ar_1_process)
plt.title('Stationary AR(1) Process', fontsize=18)
plt.show()
perform_adf_test(ar_1_process)


# Non-Stationary
ar_1_process_unit_root = generate_ar_process(1, [1], 100)
plt.figure(figsize=(10, 4))
plt.plot(ar_1_process_unit_root)
plt.title('Non-Stationary AR(1) Process', fontsize=18)
plt.show()
perform_adf_test(ar_1_process_unit_root)


# AR(2) Process
# Stationary
ar_2_process = generate_ar_process(2, [.5, .3], 100)
plt.figure(figsize=(10, 4))
plt.plot(ar_2_process)
plt.title('Stationary AR(2) Process', fontsize=18)
plt.show()
perform_adf_test(ar_2_process)


# Non-Stationary
ar_2_process_unit_root = generate_ar_process(2, [.7, .3], 100)
plt.figure(figsize=(10, 4))
plt.plot(ar_2_process_unit_root)
plt.title('Non-Stationary AR(2) Process', fontsize=18)
plt.show()
perform_adf_test(ar_2_process_unit_root)

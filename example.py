import matplotlib.pyplot as plt
import numpy as np

## STATIONARY GAUSSIAN GENERATION
from SignalForge.stationary_gaussian import StationaryGaussian

fpsd = [10,20]  # Frequency vector [Hz]
psd = [1,1]     # Power density vecrtor [(m/s^2)^2/Hz]
fs = 200        # Sampling frequency [Hz]
T = 200         # Time duration [s]
gauss_signal = StationaryGaussian(
    fpsd=fpsd, 
    psd=psd, 
    fs=fs, 
    T = T, 
    name = 'Stationary Gaussian', 
    unit='$m/s^2$'
    ) # StationaryGaussian class initialization
print(gauss_signal)
gauss_signal.plot()           # Plotting Timehistory
gauss_signal.plot_psd()       # Plotting Power spectral density 

## STATIONARY NONGAUSSIAN GENERATION
from SignalForge.stationary_nongaussian import StationaryNonGaussian

fpsd = [10,20]  # Frequency vector [Hz]
psd = [1,1]     # Power density vecrtor [(m/s^2)^2/Hz]
fs = 200        # Sampling frequency [Hz]
T = 200         # Time duration [s]
input_kurtosis = 6
nongauss_signal = StationaryNonGaussian(
    fpsd = fpsd, 
    psd = psd,
    T = T, 
    kurtosis=input_kurtosis,
    fs = fs, 
    method='zmnl', 
    name = 'Stationary NonGaussian', 
    unit='$m/s^2$'
    ) # StationaryNonGaussian class initalization
print(nongauss_signal)
nongauss_signal.plot()      # Plotting Timehistory
nongauss_signal.plot_psd()  # Plotting Power spectral density 
nongauss_signal.plot_fft()  # Plotting Fourier Transform

## NONSTATIONARY NONGAUSSIAN GENERATION
from SignalForge.nonstationary_nongaussian import NonStationaryNonGaussian

fpsd = [10,20]  # Frequency vector [Hz]
psd = [1,1]     # Power density vecrtor [(m/s^2)^2/Hz]
fs = 200        # Sampling frequency [Hz]
T = 200         # Time duration [s]
params = {'input_kurtosis': 10}
nonstat_signal = NonStationaryNonGaussian(
    fpsd = fpsd, 
    psd = psd, 
    T = T,
    params = params,
    fs = fs, 
    method='trapp_am', 
    name = 'NonStationary NonGaussian', 
    unit='$m/s^2$'
    ) # NonStationaryNonGaussian class initalization
print(nonstat_signal)
nonstat_signal.plot()      # Plotting Timehistory
nonstat_signal.plot_psd()  # Plotting Timehistory
nonstat_signal.plot_sftf()  # Plotting Short Time Fourier Transform

plt.show(block = True)
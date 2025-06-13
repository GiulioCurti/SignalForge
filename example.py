## STATIONARY GAUSSIAN GENERATION
from SignalForge import StationaryGaussian

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

gauss_signal.plot()           # Plotting Timehistory
gauss_signal.plot_psd()       # Plotting Power spectral density 

## STATIONARY NONGAUSSIAN GENERATION
from SignalForge import StationaryNonGaussian

fpsd = [10,20]  # Frequency vector [Hz]
psd = [1,1]     # Power density vector [(m/s^2)^2/Hz]
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

nongauss_signal.plot()      # Plotting Timehistory
nongauss_signal.plot_psd()  # Plotting Power spectral density 
nongauss_signal.plot_fft()  # Plotting Fourier Transform
nongauss_signal.transform_params

## NONSTATIONARY NONGAUSSIAN GENERATION AMPLITUDE MODULATED
from SignalForge import NonStationaryNonGaussian

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
    name = 'NonStationary NonGaussian AM', 
    unit='$m/s^2$'
    ) # NonStationaryNonGaussian class initalization

nonstat_signal.plot()      # Plotting Timehistory
nonstat_signal.plot_psd()  # Plotting Timehistory
nonstat_signal.plot_stft()  # Plotting Short Time Fourier Transform

## NONSTATIONARY NONGAUSSIAN GENERATION FREQUENCY MODULATED

from SignalForge import NonStationaryNonGaussian
import numpy as np

fpsd = [10,20]  # Frequency vector [Hz]
psd = [1,1]     # Power density vecrtor [(m/s^2)^2/Hz]
fs = 200        # Sampling frequency [Hz]
T = 200         # Time duration [s]

t = np.linspace(0,T,round(T*fs)) # Time samples vector 
central_impulse = 100            # Parameter for the modulation function
var = 0.5                        # Parameter for the modulation function
modulation_function = np.abs(60*np.sinc(1/T*2*np.pi*(t-central_impulse)))
params = modulation_function 

nonstat_signal = NonStationaryNonGaussian(
    fpsd = fpsd,
    psd = psd,
    fs = fs,
    T = T,
    method = 'fm',
    params = params,
    name = 'NonStationary NonGaussian FM',
    unit = 'g'
)

nonstat_signal.plot()                        # Plotting Timehistory
nonstat_signal.plot_psd()                    # Plotting Timehistory
nonstat_signal.plot_stft()                   # Plotting Short Time Fourier Transform
nonstat_signal.plot_spectral_kurtosis()      # Plotting Timehistory

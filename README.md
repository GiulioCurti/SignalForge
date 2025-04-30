SignalForge
-----------

SignalForge is an open-source Python package that provides a comprehensive set of tools for analyzing and generating non-Gaussian, non-stationary signals in engineering applications. While traditional signal processing methods often rely on assumptions of Gaussianity and stationarity, real-world signals—especially those found in mechanical systems—frequently deviate from these idealized models. SignalForge bridges this gap by offering advanced, yet user-friendly tools to simulate, inspect, and manipulate signals with complex statistical properties. Whether you're designing loads for testing, exploring time-varying behavior, or developing new algorithms, SignalForge enables you to work with signals that better reflect real operating conditions. Built with extensibility and clarity in mind, the package is suitable for researchers, practitioners, and students working across mechanical engineering, signal processing, and beyond.

Example code on how to generate a Stationary Gaussian Signal
------------------------------------------------------------

Stationary Gaussian signals can be easily generated via the specific class StationaryGaussian. Its usage is briefly shown in the following example code:

.. code-block:: python

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

Example code on how to generate a Stationary Non Gaussian Signal
------------------------------------------------------------

Stationary non Gaussian signals can be easily generated via the specific class StationaryNonGaussian. Its usage is briefly shown in the following example code:

.. code-block:: python

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


Example code on how to generate a Non Stationary Non Gaussian Signal
------------------------------------------------------------

Non stationary non Gaussian signals can be easily generated via the specific class NonStationaryGaussian. Its usage is briefly shown in the following example code:

.. code-block:: python

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
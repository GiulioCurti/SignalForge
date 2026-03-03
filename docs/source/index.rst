.. SignalForge documentation master file, created by
   sphinx-quickstart on Wed Jun 25 12:07:40 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SignalForge documentation
=========================

SignalForge is an open-source Python package that provides a comprehensive set of tools for analyzing and generating non-Gaussian, non-stationary signals in engineering applications. While traditional signal processing methods often rely on assumptions of Gaussianity and stationarity, real-world signals—especially those found in mechanical systems—frequently deviate from these idealized models. SignalForge bridges this gap by offering advanced, yet user-friendly tools to simulate, inspect, and manipulate signals with complex statistical properties. Whether you're designing loads for testing, exploring time-varying behavior, or developing new algorithms, SignalForge enables you to work with signals that better reflect real operating conditions. Built with extensibility and clarity in mind, the package is suitable for researchers, practitioners, and students working across mechanical engineering, signal processing, and beyond.

Installation
------------

SignalForge is available on PyPI. To install it, run:

.. code-block:: bash

   pip install signalforge

Alternatively, you can clone the repository from GitHub and install it in editable mode:

.. code-block:: bash

   git clone https://github.com/GiulioCurti/SignalForge.git
   cd SignalForge
   pip install -e .

Quick Start Examples
--------------------

Generating a Stationary Gaussian Signal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stationary Gaussian signals can be generated using the :class:`StationaryGaussian` class.

.. code-block:: python

   from SignalForge import StationaryGaussian

   fpsd = [10, 20]  # Frequency vector [Hz]
   psd = [1, 1]     # Power density vector [(m/s^2)^2/Hz]
   fs = 200         # Sampling frequency [Hz]
   T = 200          # Time duration [s]

   gauss_signal = StationaryGaussian(
       fpsd=fpsd, 
       psd=psd, 
       fs=fs, 
       T=T, 
       name='Stationary Gaussian', 
       unit='$m/s^2$'
   )

   gauss_signal.plot()      # Plotting Time history
   gauss_signal.plot_psd()  # Plotting Power Spectral Density

Generating a Stationary Non-Gaussian Signal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To generate stationary non-Gaussian signals with a target kurtosis, use the :class:`StationaryNonGaussian` class.

.. code-block:: python

   from SignalForge import StationaryNonGaussian

   fpsd = [10, 20]  # Frequency vector [Hz]
   psd = [1, 1]     # Power density vector [(m/s^2)^2/Hz]
   fs = 200         # Sampling frequency [Hz]
   T = 200          # Time duration [s]
   input_kurtosis = 6

   nongauss_signal = StationaryNonGaussian(
       fpsd=fpsd, 
       psd=psd,
       T=T, 
       kurtosis=input_kurtosis,
       fs=fs, 
       method='zmnl', 
       name='Stationary NonGaussian', 
       unit='$m/s^2$'
   )

   nongauss_signal.plot()      # Plotting Time history
   nongauss_signal.plot_psd()  # Plotting Power Spectral Density

Generating a Non-Stationary Non-Gaussian Signal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For non-stationary behaviors (e.g., amplitude or frequency modulation), use the :class:`NonStationaryNonGaussian` class.

.. code-block:: python

   from SignalForge import NonStationaryNonGaussian

   fpsd = [10, 20]  # Frequency vector [Hz]
   psd = [1, 1]     # Power density vector [(m/s^2)^2/Hz]
   fs = 200         # Sampling frequency [Hz]
   T = 200          # Time duration [s]
   params = {'input_kurtosis': 10}

   nonstat_signal = NonStationaryNonGaussian(
       fpsd=fpsd, 
       psd=psd, 
       T=T,
       params=params,
       fs=fs, 
       method='trapp_am', 
       name='NonStationary NonGaussian', 
       unit='$m/s^2$'
   )

   nonstat_signal.plot()       # Plotting Time history
   nonstat_signal.plot_stft()   # Plotting Short Time Fourier Transform

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
   




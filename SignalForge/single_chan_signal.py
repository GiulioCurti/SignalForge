import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pywt
import os


from .utils import *


class SingleChanSignal:
    """
    SingleChanSignal is the parent class of StationaryGaussian, StationaryNonGaussian and NonStationaryNonGaussian.
    Advanced signal analisys class.
    
    Parameters
    ----------
    x: np.ndarray
        frequency vector of the power spectral density
    fs: float
        power density vector of the power spectral density
    dfpsd: float
        frequency discretization of the psd to be stored 
    name: str 
        name of the process (used for plots)
    var: str
        name of the variable (used for plots)
    unit: str 
        unit of measure of the signal (used for plots)
    seed: int
        seed for the random generator used in non-gaussian models
    interp: 'lin' or 'log'
        interpolation rule for increasing resolution on given PSD     
    """
    
    def __init__(
        self, x:np.ndarray,
        fs:float, dfpsd = 0.5, 
        name:str = "", 
        var:str = 'x', 
        unit:str = "$m/s^2$",
        signal_type:str = 'User provided signal'
    ):
        
        self.name = name
        self.var = var
        self.unit = unit        
        self.signal_type = signal_type
            
        print('Estimating statistical parameters from timehistory')
        self.x = np.matrix.flatten(np.array(x))
        self.mean = np.mean(self.x)
        self.fs = float(fs)
        self.N = len(self.x)
        self.T = self.N/self.fs
        self.dt = 1/self.fs
        self.dfpsd = dfpsd
        self.fpsd,self.psd = self.get_psd(dfpsd)
        
        self.central_moments = self._get_central_moments()
        self.spectral_moments = self._get_spectral_moments()
        self.ns_index = {}
        print('Class correctly initialized')
        print(self)
        
    def __str__(self):
        """
        Human-readable summary of the signal object.
        """
        return (
            f"SingleChanSignal Summary:\n"
            f"--------------------------\n"
            f"Name         : {self.name}\n"
            f"Type         : {self.signal_type}\n"
            f"Length       : {self.N} samples\n"
            f"Duration     : {self.T:.2f} s\n"
            f"Sampling Rate: {self.fs:.2f} Hz\n"
            f"Mean         : {self.mean:.2f}\n"
            f"Std. Dev.    : {self.central_moments['rms']:.2f} {self.unit}\n"
            f"Skewness     : {self.central_moments['skew']:.2f} {self.unit}\n"
            f"Kurtosis     : {self.central_moments['kurtosis']:.2f}\n"
            f"Crest factor : {self.central_moments['crest_factor']:.2f}\n"
            f"0-cross rate : {self.spectral_moments['v0']:.2f} Hz\n"
            f"Peak rate    : {self.spectral_moments['vp']:.2f} Hz"
        )
    
    def __repr__(self):
        """
        Return a developer-friendly string representation of the object.
        Includes key attributes for quick inspection in interactive sessions.
        """
        return (
            f"<SingleChanSignal | "
            f"N={self.N}, fs={self.fs:.2f} Hz, T={self.T:.3f} s, "
            f"mean={np.mean(self.x):.3f}, std={np.std(self.x):.3f}, "
            f"min={np.min(self.x):.3f}, max={np.max(self.x):.3f}>"
        )
    
    def __add__(self, other):
        """
        Overload the + operator to allow addition of:
        - Two SingleChanSignal objects (must have same length, dt, and aligned timebase).
        - A SingleChanSignal and a scalar (adds scalar to each sample).

        Returns
        -------
        - A new SingleChanSignal instance.

        Raises
        ------
        - TypeError: If the operand type is unsupported.
        - ValueError: If adding two signals of different lengths or mismatched time bases.
        """
        if isinstance(other, SingleChanSignal):
            if self.N != other.N:
                raise ValueError("Cannot add signals: different lengths.")
            if not np.isclose(self.dt, other.dt):
                raise ValueError("Cannot add signals: different sampling intervals.")
            if not np.allclose(self.t, other.t):
                raise ValueError("Cannot add signals: time vectors are not aligned.")

            return SingleChanSignal(x=self.x + other.x, fs=self.fs)

        elif isinstance(other, (int, float)):
            return SingleChanSignal(x=self.x + other, fs=self.fs)

        else:
            raise TypeError(f"Unsupported operand type(s) for +: 'SingleChanSignal' and '{type(other).__name__}'")
        
    
    @property # for storage efficiency
    def t(self):
        """time vector"""                                       
        return np.linspace(0,self.T,self.N)
    
    @property # for storage efficiency
    def fft_ts(self):
        """two-sided Fourier coefficients"""   
        Xts_woShift = self.dt*np.fft.fft(self.x)
        return np.fft.fftshift(Xts_woShift)
    
    @property # for storage efficiency
    def f_fft_ts(self):
        """two-sided frequency vector"""
        frq_woShift = np.fft.fftfreq(self.N, self.dt)
        return np.fft.fftshift(frq_woShift) 
    
    @property # for storage efficiency
    def fft_os(self):
        """one-sided Fourier coefficients"""
        Xts_woShift = self.dt*np.fft.fft(self.x)
        Nos         = np.ceil((self.N+1)/2)             # number of one-sided Fourier coefficients
        return Xts_woShift[0:int(Nos)]       
     
    @property # for storage efficiency
    def f_fft_os(self):
        """one-sided frequency vector"""
        frq_woShift = np.fft.fftfreq(self.N, self.dt)
        Nos         = np.ceil((self.N+1)/2)             # number of one-sided Fourier coefficients
        return np.abs(frq_woShift[0:int(Nos)])
    

    def get_psd(self,df = 0.5,noverlap=None):
        ''' 
        estimate power spectral density (PSD) using the Welch method 
        -> called by __init__
        '''
        self.fpsd, self.psd = sp.signal.welch(self.x,self.fs,nperseg=np.round(self.fs/df), noverlap = noverlap)
        return self.fpsd, self.psd 
    
    def _get_central_moments(self):
        ''' 
        estimate key statistical values 
        -> called by __init__     
        SOURCES
        -------
            A.G. Davenport, Note on the distribution of the largest value of a random function with application to gust loading, Proceedings of the Institution of Civil Engineers 28 (2) (1964) 187–196
        '''    
        cmoms = {'rms': np.std(self.x)}
        cmoms['var'] = np.var(self.x)
        cmoms['skew'] = sp.stats.skew(self.x)
        cmoms['kurtosis'] = sp.stats.kurtosis(self.x)+3
        cmoms['crest_factor'] = np.max(np.abs(self.x)) / cmoms['rms']
        return cmoms
    
    def _get_spectral_moments(self):
        ''' 
        estimate key spectral descriptors 
        -> called by __init__     
        SOURCES
        -------
            C. L. Nikias e A. P. Petropulu, Higher-order spectra analysis: a nonlinear signal processing framework. in Prentice Hall signal processing series. Englewood Cliffs, N.J: PTR Prentice Hall, 1993
        '''   
        smoms = {'smom0': get_stat_mom(self.fpsd, self.psd, 0), 
                'smom1': get_stat_mom(self.fpsd, self.psd, 1),
                'smom2': get_stat_mom(self.fpsd, self.psd, 2),
                'smom4': get_stat_mom(self.fpsd, self.psd, 4),
                'smom6': get_stat_mom(self.fpsd, self.psd, 6)}
        smoms['v0']   = np.sqrt(smoms['smom2']/smoms['smom0'])
        smoms['vp']   = np.sqrt(smoms['smom4']/smoms['smom2'])
        smoms['vm']   = smoms['smom1']/smoms['smom0']
        smoms['xm'] = smoms['vm']/ smoms['vp']
        smoms['alpha1'] = alpha_spec_index(self.fpsd, self.psd, 1, 0)
        smoms['alpha2'] = alpha_spec_index(self.fpsd, self.psd, 2, 0)
        smoms['alpha075'] = alpha_spec_index(self.fpsd, self.psd, .75, 0)
        smoms['alphadx'] = alpha_spec_index(self.fpsd, self.psd, 2, 1)
        return smoms
    
    def get_pdf(self, bins='auto'):
        """
        Estimate the empirical probability density function (PDF) of the signal.

        Parameters
        ----------
        bins : int, sequence, or str, optional
            Binning strategy for histogram computation. Passed directly to `np.histogram`.

        Returns
        -------
        grid : ndarray
            Center values of the histogram bins.
        prob : ndarray
            Normalized histogram values representing the PDF.
        """
        prob, bin_edges = np.histogram(self.x, bins = bins, density=True)
        grid = (bin_edges[1:]+bin_edges[0:-1])/2
        return  grid,prob
    
    def get_bestfit_gaussian_pdf(self,bins = 'auto', grid = None):
        """
        Compute the best-fit Gaussian PDF based on the signal's mean and variance.

        Parameters
        ----------
        bins : int, sequence, or str, optional
            Used to compute a default grid if one is not provided.
        grid : ndarray, optional
            The grid (support) over which to evaluate the Gaussian PDF. If None, computed from histogram.

        Returns
        -------
        grid : ndarray
            Grid points where the Gaussian PDF is evaluated.
        pgauss : ndarray
            Gaussian PDF values on the grid.
        """
        if grid is None:
            _, bin_edges = np.histogram(self.x, bins, density=True)
            grid = (bin_edges[1:]+bin_edges[0:-1])/2
        
        # Compute Gaussian PDF using the sample mean and variance    
        pgauss = gaussian_bell(grid = grid, var = self.central_moments['var'], mean = self.mean)
        return grid, pgauss
    
    def get_sftf(self, nperseg:int = 2**10, hop = 2**5, nargout = 1):
        """
        Compute the Short-Time Fourier Transform (STFT) of the signal.

        Parameters
        ----------
        nperseg : int, optional
            Length of each segment (window size) for the STFT.
        hop : int, optional
            Hop size (step between windows).
        nargout : int, optional
            If >1, also returns the SFTF object.

        Returns
        -------
        Sx : ndarray
            STFT magnitude spectrum.
        SFT : ShortTimeFFT, optional
            SFTF object containing additional metadata and transform options.
        """
        w = sp.signal.windows.hann(nperseg)  
        SFT = sp.signal.ShortTimeFFT(w, hop=hop, fs=self.fs, mfft=nperseg*2, scale_to='magnitude')
        Sx = SFT.stft(self.x)
        if nargout>1:
            return Sx, SFT    
        return Sx
    
    def get_spectrogram(self, nperseg:int = 2**10, hop = 2**5, nargout = 1):
        """
        Compute the Spectrogram of the signal via Short-Time Fourier Transform (STFT).

        Parameters
        ----------
        nperseg : int, optional
            Length of each segment (window size) for the STFT.
        hop : int, optional
            Hop size (step between windows).
        nargout : int, optional
            If >1, also returns the SFTF object.

        Returns
        -------
        Sxx : ndarray
            STFT power densitt spectrum.
        SFT : ShortTimeFFT, optional
            SFTF object containing additional metadata and transform options.
        """ 
        w = sp.signal.windows.hann(nperseg)  
        SFT = sp.signal.ShortTimeFFT(w, hop=hop, fs=self.fs, mfft=nperseg*2, scale_to='psd')
        Sxx = SFT.stft(self.x)
        if nargout>1:
            return Sxx, SFT    
        return Sxx  
        
    def get_hilbert(self, *args, **kwargs):
        """
        Compute the analytic signal using the Hilbert transform.

        Parameters
        ----------
        *args, **kwargs : optional
            Additional arguments passed to `scipy.signal.hilbert`.

        Returns
        -------
        analytic_signal : ndarray
            Complex-valued analytic signal.
        amplitude_envelope : ndarray
            Instantaneous amplitude (envelope) of the signal.
        instantaneous_phase : ndarray
            Unwrapped instantaneous phase of the signal.
        """
        analytic_signal = sp.signal.hilbert(self.x, *args, **kwargs)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        return analytic_signal, amplitude_envelope, instantaneous_phase
    
    def get_wavelet_transform(self, wavetype = 'morl', **kwargs):
        """
        Compute the Continuous Wavelet Transform (CWT) of the signal.

        Parameters
        ----------
        wavetype : str, optional
            Type of mother wavelet to use (e.g., 'morl', 'cmor').
        **kwargs : dict, optional
            Additional keyword arguments passed to `pywt.cwt`.

        Returns
        -------
        data : ndarray
            Wavelet coefficient matrix.
        f : ndarray
            Frequencies corresponding to wavelet scales.
        """
        print(f'Calculating continuyous wavelet transform with "{wavetype}" wave')
        wavelet = pywt.ContinuousWavelet(wavetype)
        scales = pywt.central_frequency(wavelet) * self.fs / np.arange(10, 150, 1)
        data, f =  pywt.cwt(self.x, scales, wavelet, sampling_period= 1/self.fs, **kwargs)
        return data, f
    
    def get_banded_spectral_kurtosis(self, n_bins = 10, fl : int = 0, fu : int = 0):
        """
        Compute banded spectral kurtosis over specified frequency bands.

        Parameters
        ----------
        n_bins : int, optional
            Number of frequency bands to divide the spectrum into.
        fl : int, optional
            Lower frequency limit (Hz).
        fu : int, optional
            Upper frequency limit (Hz).

        Returns
        -------
        fvec : ndarray
            Center frequencies of each band.
        SK : ndarray
            Spectral kurtosis values for each frequency band.
        """
        fvec, SK = get_banded_spectral_kurtosis(self.x, self.dt, n_bins = n_bins, fl = fl, fu = fu) 
        return fvec, SK
    
    def get_welch_spectral_kurtosis(self, Nfft:int = None, noverlap:int = 0):
        """
        Compute Welch-based spectral kurtosis of the signal.

        Parameters
        ----------
        Nfft : int, optional
            FFT length used in the Welch method. Defaults to length of `self.psd`.
        noverlap : int, optional
            Number of overlapping points between segments.

        Returns
        -------
        f_norm : ndarray
            Frequency vector (in Hz).
        SK : ndarray
            Spectral kurtosis values across the frequency range.
        """
        if Nfft is None: 
            Nfft = len(self.psd)
        
        f_norm, SK = get_welch_spectral_kurtosis(self.x, Nfft, noverlap)
        return f_norm*self.fs, SK

    def get_kurtogram(self, n_bins:tuple = (10,100,10), _plot_call = False):
        """
        Calculate the kurtogram of the signal.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, a new figure and axes are created.
        n_bins : tuple, optional
            The range of bins to use for the kurtogram. (start, finish, step)

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the plotted kurtogram.
        """
        print(f'Calculating Kurtogram')
        np_freqs = self.N//2
        try:
            n_divisions = np.arange(*n_bins)
        except:
            raise ValueError('n_bins argument does not meet requirement. Try with a tuple with np.arange() format "(start, finish, step)"')
        max_Nfft = np_freqs//np.min(n_divisions)
        ref_fvec = np.linspace(0,1,max_Nfft)*self.fs/2
        kurtogram = np.zeros((max_Nfft,len(n_divisions)))
        for i, divisions in tqdm(enumerate(n_divisions)):
            Nfft = np_freqs//(divisions)
            fvec, spectral_kurtosis = get_welch_spectral_kurtosis(self.x,Nfft = Nfft, noverlap = round(3/8*Nfft))
            kurtogram[:,i] = step_interp(ref_fvec, fvec*self.fs, spectral_kurtosis)
        if _plot_call:
            return kurtogram, n_divisions, max_Nfft
        else:
            return kurtogram
        
    def get_stat_history(self, winsize:int, olap:int = 0, idx_type = 'rms'):
        """
        Compute a statistical index over a moving window.

        Parameters:
            signal : np.ndarray
                Time history signal.
            winsize : int: 
                Window size.
            olap : int, optional 
                The number of points to overlap on the moving window. If not specified, no overlap is considered
            idx_type : str 
                Index type: 'mean', 'rms', 'var', or 'kurtosis'.

        Returns:
            stat_history: np.ndarray
                Evolution of the statistical index over time.
            
        """        
        stat_th = get_stat_history(self.x, winsize=winsize, olap = olap, idx_type = idx_type)
        return stat_th
    
    def get_nonstat_index(self, idx_type = 'nnst',*args, **kwargs):
        """
        Computes the requested non stationary test from the given one dimensional time history

        Parameters
        ---------- 
        signal: array-like
            One dimensional time history
        index_type : str, optional
            Type of non stationary index to be assessed. Supported methods are: Non-Stationarity index [1]: 'nnst',  Augmented Dickey-Fuller [2]: 'adf', Kwiatkowski-Phillips-Schmidt-Shin [3]: 'kpss'
        kwargs: optional
            Argument to be passed to the specific method implementation
        
        Returns
        -------
        test_results: dict
            Results of the non-stationary test    
        
        Source
        ------
        [1] L. Capponi, M. Česnik, J. Slavič, F. Cianetti, e M. Boltežar, «Non-stationarity index in vibration fatigue: Theoretical and experimental research», Int. J. Fatigue, vol. 104, pp. 221–230, nov. 2017, doi: 10.1016/j.ijfatigue.2017.07.020.
        [2] Dickey, D. A., and W. A. Fuller. "Distribution of the Estimators for Autoregressive Time Series with a Unit Root." Journal of the American Statistical Association. Vol. 74, 1979, pp. 427–431.
        [3] Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., Shin, Y. (1992). "Testing the null hypothesis of stationarity against the alternative of a unit root". Journal of Econometrics. 54 (1–3): 159–178. doi:10.1016/0304-4076(92)90104-Y.

        """
        # ns_index = get_nonstat_test(self.x, idx_type, *args, **kwargs)
        idx_types = {'nnst': get_nnst_index,
                    'adf': get_adf_index,
                    'kpss': get_kpss_index}
    
        method_existance_check(idx_type, idx_types)

        ns_index = idx_types[idx_type](self.x, *args, **kwargs)
        print(f'Nonstat evaluation via "{idx_type}" method:')
        print_nonstat_results(ns_index, indent=4)
        self.ns_index[idx_type] = ns_index
        return ns_index
    
    def plot(
        self, ax = None, 
        xlims:list = None, 
        ylims:list= None
        ):
        """
        Plot the time history of the signal.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, a new figure and axes are created.
        xlims : list, optional 
            The limits for the x-axis.
        ylims : list, optional 
            The limits for the y-axis.

        Returns
        -------
        ax :  matplotlib.axes.Axes 
            The axes with the plotted time history.
        """
        print(f'Plotting Timehistory')
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        s_rms = self.central_moments['rms']
        s_skew = self.central_moments['skew']
        s_kurt = self.central_moments['kurtosis']
        label = f"{self.name}: rms = {s_rms:.2f} [{self.unit}] | $skew $ = {s_skew:.2f} [-] | $kurt $ = {s_kurt:.2f} [-]"
        ax.plot(self.t,self.x,'k',label=label)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"Amplitude [{self.unit}]")
        ax.legend()   
        ax.grid(True,which = 'both')
        ax.minorticks_on()   
        ax.set_title(f'Timehistory {self.name}')
        if xlims is not None:
            ax.set_xlim(xlims)
        if ylims is not None:
            ax.set_ylim(ylims)
        plt.show(block=False)
        return ax
    
    def plot_pdf(
        self, 
        ax = None, 
        xlims:list = None, 
        log:bool = False, 
        plot_gaussian_bestfit = True):
        
        """
        Plot the Probability Density Function (PDF) of the signal.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional 
            The axes to plot on. If None, a new figure and axes are created.
        xlims : list, optional 
            The limits for the x-axis.
        log : bool, optional  
            Whether to plot on a logarithmic scale.
        plot_gaussian_bestfit: bool, optional 
            Whether to plot the best-fit Gaussian distribution.

        Returns
        -------
        ax : matplotlib.axes.Axes 
            The axes with the plotted PDF.
        """
        print(f"Plotting Probability densisy function of signal's points")
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))
        # s_rms = self.central_moments['rms']
        s_kurt = self.central_moments['kurtosis']
        grid, pdf = self.get_pdf()
        
        label = f"{self.name}: $kurt$ = {s_kurt:.2f} [-] "
        
        ax.plot(grid, pdf,'k',label=label)
        
        grid_gauss, pdf_gauss = self.get_bestfit_gaussian_pdf(grid=grid) # estimating gaussian counterpart for limits
        if plot_gaussian_bestfit:
            ax.plot(grid_gauss, pdf_gauss, 'r', label="Best-fit Gaussian")
            
        ax.set_xlabel(f"Amplitude [{self.unit}]")
        ax.set_ylabel(f"$PDF$ [-]")
        ax.legend()   
        ax.grid(True,which = 'both')
        ax.minorticks_on()   
        ax.set_title(f'Probability Density Function {self.name}')
        ax.set_ylim([np.min(pdf_gauss), 1.1*np.max([np.max(pdf_gauss), np.max(pdf)])])
        if xlims is not None:
            ax.set_xlim(xlims)
        if log:
            ax.set_yscale("log") 
        plt.show(block=False)
        return ax
    
    def plot_fft(
        self,
        ax = None, 
        xlims:list = None, 
        ylims:list= None
        ):
        """
        Plot the Fourier Transform (FT) of the signal.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, a new figure and axes are created.
        xlims : list, optional
            The limits for the x-axis.
        ylims : list, optional 
            The limits for the y-axis.

        Returns
        -------
        ax : matplotlib.axes.Axes 
            The axes with the plotted FT.
        """
        print(f'Plotting Fourier Transform of the signal')
        
        if ax is None:
            _, ax = plt.subplots(nrows=2, figsize=(10, 4), sharex='all', tight_layout=True)
        
        s_rms = self.central_moments['rms']
        label = f"$FT_{{self.var}}$: rms = {s_rms:.2f} [{self.unit}]"
        ax[0].plot(self.f_fft_os,np.abs(self.fft_os),'k',label=label)
        # ax[0].set_xlabel("Frequency [Hz]")
        ax[0].set_ylabel(f"$FT_{self.var}$ [{self.unit}]")
        ax[0].legend()   
        ax[0].grid(True,which = 'both')
        ax[0].minorticks_on()   
        ax[0].set_title(f'Fourier transform {self.name}')
        if len(ax)>1:
            ax[1].plot(self.f_fft_os,np.angle(self.fft_os),'k',label=label)
            ax[1].set_xlabel("Frequency [Hz]")
            ax[1].set_ylabel(f"$\\phi_{self.var}$ [rad]")
            # ax[1].legend()   
            ax[1].grid(True,which = 'both')
            ax[1].minorticks_on()   
        if xlims is not None:
            ax.set_xlim(xlims)
        if ylims is not None:
            ax.set_ylim(ylims)
        plt.show(block=False)
        return ax
    
    def plot_psd(
        self,
        ax = None, 
        xlims:list = None, 
        ylims:list= None, 
        log:bool = False
        ): 
        """
        Plot the Power Spectral Density (PSD) of the signal.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional 
            The axes to plot on. If None, a new figure and axes are created.
        xlims : list, optional 
            The limits for the x-axis.
        ylims : list, optional 
            The limits for the y-axis.
        log : str, optional 
            Specifies the logarithmic scale for x, y, or both axes.

        Returns
        -------
        ax : matplotlib.axes.Axes 
            The axes with the plotted PSD.
        """
        print(f'Plotting Power spectral density')
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4),tight_layout=True)
        s_rms = self.central_moments['rms']
        label = f"$G_{{{self.var}{self.var}}}$: rms = {s_rms:.2f} [{self.unit}]"
        ax.plot(self.fpsd,self.psd,'k',label=label)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(f"$G_{{{self.var}{self.var}}}$ [({self.unit})$^2$/Hz]")
        ax.legend()   
        ax.grid(True,which = 'both')
        ax.minorticks_on()        
        ax.set_title(f'Power Spectral Density {self.name}')
        if xlims is not None:
            ax.set_xlim(xlims)
        if ylims is not None:
            ax.set_ylim(ylims)
        if log == "x":
            ax.set_xscale("log")
        elif log == "y":
            ax.set_yscale("log")
        elif log == "both":
            ax.set_xscale("log")
            ax.set_yscale("log")
        plt.show(block=False)
        return ax
    
    def plot_spectrogram(
        self,
        ax = None, 
        window:tuple = ('tukey', 1), 
        nperseg:int = 2**10, 
        ylims:list = None, 
        clims:list = None
        ):
        """
        Plot the spectrogram of the signal.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional 
            The axes to plot on. If None, a new figure and axes are created.
        window : tuple, optional 
            The window type and parameter for the spectrogram.
        nperseg : int, optional 
            The number of data points per segment for the spectrogram.
        ylims : list, optional 
            The limits for the y-axis.
        clims : list, optional 
            The color limits for the spectrogram.

        Returns
        -------
        ax : matplotlib.axes.Axes 
            The axes with the plotted spectrogram.
        """
        print(f'Calculating spectrogram')
        f, t, Sxx = sp.signal.spectrogram(self.x, fs = self.fs, window = window, nperseg = nperseg)       
        print(f'Plotting spectrogram')
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True)
        spec = ax.pcolormesh(t, f, Sxx, vmin=Sxx.min(), vmax=Sxx.max(), shading='auto')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [sec]')
        ax.set_title('Spectrogram')
        cbar = fig.colorbar(spec)
        cbar.ax.set_ylabel('PSD [(m/$s^2$)$^2$/Hz]')
        if clims is not None:
            spec.set_clim(clims)
        else:
            spec.set_clim(0,1.1*np.max(self.psd))
        if ylims is not None:
            ax.set_ylim(ylims)
        plt.show(block=False)
        return ax
    
    def plot_stft(
        self, 
        nperseg:int = 2**10, 
        hop = 2**8,
        flims:list = None):
        """
        Plot the Short-Time Fourier Transform (STFT) of the signal.

        Parameters
        ----------
        nperseg : int, optional
            The number of data points per segment for the STFT.
        hop : int, optional 
            The number of data points between segments.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the plotted STFT.
        """
        print(f'Calculating STFT')
        Sx, SFT = self.get_sftf(nperseg = nperseg, hop = hop, nargout=2)
        print(f'Plotting STFT')
        fig, ax = plt.subplots(figsize=(10, 4), sharex='all', tight_layout=True)
        
        t_lo, t_hi = SFT.extent(self.N)[:2]  # time range of plot
        ax.set_title(rf"Short time Fourier transform {self.name}")
        ax.set(xlabel=f"Time [s] ({SFT.p_num(self.N)} slices, " +
                    rf"$\Delta t = {SFT.delta_t:.3f}\,$s)",
                ylabel=f"Frequency [Hz] ({SFT.f_pts} bins, " +
                    rf"$\Delta f = {SFT.delta_f:.3f}\,$Hz)",
                xlim=(t_lo, t_hi))

        im1 = ax.imshow(abs(Sx), origin='lower', aspect='auto',
                        extent=SFT.extent(self.N))
        fig.colorbar(im1, label=f"Magnitude $|S_{self.var}(t, f)|$ [{self.unit}]")
        if flims:
            plt.ylim(flims)
        plt.show(block=False)

        return ax
    
    def plot_spectral_kurtosis(
        self,
        ax = None, 
        n_bins:int = None, 
        fl : int = None, 
        fu : int = None, 
        ylims:list = None
        ):
        """
        Plot the spectral kurtosis of the signal.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional 
            The axes to plot on. If None, a new figure and axes are created.
        n_bins : int, optional 
            The number of bins for the spectral kurtosis calculation.
        fl : int, optional 
            The lower frequency bound.
        fu : int, optional 
            The upper frequency bound.
        ylims : list, optional 
            The limits for the y-axis.

        Returns
        -------
        ax : matplotlib.axes.Axes 
            The axes with the plotted spectral kurtosis.
        """
        if n_bins and fl and fu: 
            fvec, spectral_kurtosis = get_banded_spectral_kurtosis(self.x,self.dt,n_bins, fl, fu)
        else: 
            fvec, spectral_kurtosis = get_welch_spectral_kurtosis(self.x,Nfft = 2**10, noverlap = 2**8)
            
        print(f'Plotting spectral kurtosis from {fvec[0]:.1f} Hz to {fvec[-1]*self.fs/2:.1f} Hz')
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        s_kur = self.central_moments['kurtosis']
        glob = 'global'
        label = f"$SK_{self.var}$(f): $kurt_{{{glob}}}$ = {s_kur:.2f} [-]"
        ax.step(fvec*self.fs/2, spectral_kurtosis,'k',label=label, where = 'mid')
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel(f"$SK_{self.var}$ [-]")
        ax.legend()   
        ax.grid(True,which = 'minor')
        ax.minorticks_on()  
        ax.set_title('Spectral kurtosis ' + self.name + ' (0 = Gaussian)')
        ax.set_ylim([0,1.5*np.max(spectral_kurtosis)])
        if ylims is not None:
            ax.set_ylim(ylims)
        plt.show(block=False)
        return ax
    
    def plot_kurtogram(
        self,
        ax = None, 
        n_bins:tuple = (10,100,10), 
        ylims:list = None
        ):
        """
        Plot the kurtogram of the signal.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, a new figure and axes are created.
        n_bins : tuple, optional
            The range of bins to use for the kurtogram. (start, finish, step)
        ylims : list, optional 
            The limits for the y-axis.

        Returns
        -------
        ax : matplotlib.axes.Axes 
            The axes with the plotted kurtogram.
        """      
        kurtogram, n_divisions, max_Nfft = self.get_kurtogram(n_bins = n_bins, _plot_call = True)
        print(f'Plotting Kurtogram')
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True)
        freq_edges = np.linspace(0, self.fs / 2, max_Nfft + 1)
        band_edges = np.append(n_divisions, n_divisions[-1] + (n_divisions[1] - n_divisions[0]))
        FE, BE = np.meshgrid(np.flip(band_edges), freq_edges)
        mesh = ax.pcolormesh(BE, FE, kurtogram, shading='auto')
        plt.gca().invert_yaxis()
        cbar = fig.colorbar(mesh, ax = ax)
        cbar.ax.set_ylabel('Spectral kurtosis')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Number of bands')
        ax.set_title(f'Kurtogram {self.name}')
        if ylims is not None:
            ax.set_ylim(ylims)
        plt.show(block=False)
        return ax    
    
    def plot_stat_history(
        self, 
        winsize:int, 
        olap:int = 0,
        idx_type:str='rms', 
        ax = None, 
        xlims:list = None, 
        ylims:list= None
        ):
        """
        Plot the statistical indicator time history of the signal on a moving window.

        Parameters
        ----------
        winsize : int 
            The window size for the non-stationarity index calculation.
        olap : int, optional
            The number of points to overlap on the moving window. If not specified, no overlap is considered
        idx_type : str, optional 
            The type of index to calculate ('rms' or 'kurt'). 
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, a new figure and axes are created.
        xlims : list, optional
            The limits for the x-axis.
        ylims : list, optional
            The limits for the y-axis.

        Returns
        -------
        ax :: matplotlib.axes.Axes 
            The axes with the plotted non-stationarity index.
        """
        ns_index = self.get_stat_history(winsize=winsize, olap = olap, idx_type = idx_type)
        print(f'Plotting Non-stationarity index: idx_type = {idx_type}')
        t_ns_index = np.linspace(0,1,len(ns_index))*self.T
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        s_rms = self.central_moments['rms']
        s_kurt = self.central_moments['kurtosis']
        label = f"{self.name}: rms = {s_rms:.2f} [{self.unit}] | $kurt $ = {s_kurt:.2f} [-]"
        ax.plot(self.t,self.x,'k',label=label, alpha = 0.5)
        ax.step(t_ns_index, ns_index, 'k', where = 'mid', label = f'{idx_type} history')
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"Amplitude [{self.unit}]")
        ax.legend()   
        ax.grid(True,which = 'both')
        ax.minorticks_on()   
        ax.set_title(f'Non-stationarity index {self.name}')
        if xlims is not None:
            ax.set_xlim(xlims)
        if ylims is not None:
            ax.set_ylim(ylims)
        plt.show(block=False)
        return ax
    
    def plot_hilbert(self, ax = None):
        """
        Plot the Hilbert transform of the signal, showing the amplitude envelope and instantaneous frequency.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional 
            The axes to plot on. If None, a new figure and axes are created.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the plotted Hilbert transform.
        """
        _, amplitude_envelope, instantaneous_phase = self.get_hilbert()
        print(f'Plotting Hilbert transform')
        instantaneous_frequency = np.diff(instantaneous_phase) / (2*np.pi) * self.fs
        time = self.t
        if ax is None:
            _, ax = plt.subplots(nrows=2, figsize=(10, 4), sharex='all', tight_layout=True)
        ax[0].set_title(f"Hilbert transform {self.name}")
        ax[0].set_ylabel(f"Amplitude [{self.unit}]")
        ax[0].plot(time, self.x, 'k', label=f'Signal')
        ax[0].plot(time, amplitude_envelope, 'r', label='Envelope')
        ax[0].legend()
        ax[0].grid(True,which = 'both')
        ax[0].minorticks_on() 
        ax[1].set(xlabel="Time [s]", ylabel="Frequency [Hz]")
        ax[1].plot(time[1:], abs(instantaneous_frequency), 'k', label = 'Instantaneous frequency')
        ax[1].legend()
        ax[1].grid(True,which = 'both')
        ax[1].minorticks_on() 
        plt.show(block=False)
        return ax
    
    def _plot_hilbert_spectrum(self, ax = None): #! WIP
        """
        Plot the Hilbert spectrum of the signal, showing the amplitude and phase.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, a new figure and axes are created.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the plotted Hilbert spectrum.
        """
        _, amplitude_envelope, instantaneous_phase = self.get_hilbert()
        instantaneous_frequency = np.diff(instantaneous_phase,prepend=0) / (2*np.pi) * self.fs
        h_spectrum =  np.fft.rfft(amplitude_envelope)/self.fs
        freq = self.f_fft_os
        if ax is None:
            _, ax = plt.subplots(nrows=2, figsize=(10, 4), sharex='all', tight_layout=True)
        ax[0].set_title(f"Hilbert spectrum {self.name}")
        ax[0].set_ylabel(f"Amplitude [{self.unit}]")
        ax[0].plot(freq, abs(h_spectrum), 'k')
        ax[0].legend()
        ax[0].grid(True,which = 'both')
        ax[0].minorticks_on() 
        ax[1].set(xlabel="Frequency [Hz]", ylabel="Phase [rad]")
        ax[1].plot(freq, np.angle(h_spectrum), 'k')
        # ax[1].legend()
        ax[1].grid(True,which = 'both')
        ax[1].minorticks_on() 
        plt.show(block=False)
        return ax
        
    def plot_scalogram(self, ax = None, wavetype = 'morl', **kwargs):
        """
        Plot the scalogram of the signal using wavelet transformation.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, a new figure and axes are created.
        wavetype : str, optional
            The type of wavelet to use for the transform.
        **kwargs: 
        Additional keyword arguments for wavelet transformation.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the plotted scalogram.
        """
        data, f = self.get_wavelet_transform(wavetype=wavetype, **kwargs)
        magnitude = np.log10(np.abs(data)+1)
        print(f'Plotting Scalogram')
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True)
        spec = ax.pcolormesh(self.t, f, magnitude, vmin=magnitude.min(), vmax=magnitude.max(), shading='auto')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [s]')
        ax.set_title(f"Scalogram with '{wavetype}' wave {self.name}")
        cbar = fig.colorbar(spec)
        cbar.ax.set_ylabel('Wavelet vector coefficiens [-]')
        plt.show(block=False)
        return ax

    def save_to_csv(self, filename: str):
        """
        Save the signal time history to a CSV file using numpy.savetxt.

        Parameters
        ----------
        filename : str
            Full path (or filename) where the CSV will be saved.
        """
        # Stack time and signal into two columns
        data = np.column_stack((self.t, self.x))

        # Ensure directory exists if specified
        if os.path.dirname(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save to CSV
        np.savetxt(filename, data, delimiter=",")

        print(f"Signal saved to '{filename}' successfully.")
        
if __name__ == "__main__":
    pass
    

    
        
        
    
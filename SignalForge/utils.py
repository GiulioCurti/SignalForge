import numpy as np
import scipy as sp
from tqdm import tqdm
import pyNNST
from statsmodels.tsa.stattools import kpss, adfuller


def method_existance_check(method:str, methods:dict):
    """
    Check if a method is in a dictionary of available options.

    Parameters:
        method (str): The method name to check.
        methods (dict): A dictionary of available methods.

    Raises:
        KeyError: If method is not in the dictionary keys.
    """
    if method not in methods.keys():
        options = ', '.join(f'"{key}"' for key in methods.keys())
        raise KeyError(f'Index {method} not supported. Try with: {options}')
        # raise KeyError(f'Index {method} not supported. Try with: {[f'"{key}"' for key in methods.keys()]}')

def step_interp(x, xp, yp):
    """
    Step interpolation of `yp` over `xp`, evaluated at `x`.

    Parameters:
        x (array-like): Points at which to evaluate the interpolated values.
        xp (array-like): Known x-coordinates of data points.
        yp (array-like): Known y-values at each xp.

    Returns:
        np.ndarray: Interpolated values at `x`.
    """
    xp = np.asarray(xp)
    yp = np.asarray(yp)
    x_mid = (xp[:-1] + xp[1:]) / 2
    x_extended = np.concatenate([[xp[0]], x_mid, [xp[-1]]])
    y_extended = np.concatenate([[yp[0]], yp[1:], [yp[-1]]])
    idx = np.searchsorted(x_extended, x, side='right') - 1
    y = y_extended
    idx = np.clip(idx, 0, len(yp)-1)
    return y[idx]

def lin_interp_psd(fpsd:np.ndarray, psd:np.ndarray, n_points:int, fs = None): 
    """
    Linear interpolation of a PSD to a uniformly spaced frequency grid.

    Parameters:
        fpsd (array-like): Original frequency vector.
        psd (array-like): Original PSD values.
        n_points (int): Number of points in the interpolated frequency vector.

    Returns:
        tuple: (interpolated frequency vector, interpolated PSD)
    """
    if fs is None: 
        fs = fpsd[-1] * 2
    new_fpsd = np.linspace(0,1,round(n_points))*fs/2
    new_psd = np.interp(new_fpsd, fpsd, psd, left=0, right=0)
    new_psd[0] = 0
    return new_fpsd, new_psd 

def log_interp_psd(fpsd, psd, n_points, fs = None):
    """
    Log-log interpolation of a PSD to a uniformly spaced frequency grid.

    Parameters:
        fpsd (array-like): Original frequency vector.
        psd (array-like): Original PSD values.
        n_points (int): Number of points in the interpolated frequency vector.

    Returns:
        tuple: (interpolated frequency vector, interpolated PSD)
    """
    if fs is None: 
        fs = fpsd[-1] * 2
    psd = psd.astype('float')
    psd[psd<=0]= 1e-30
    new_fpsd = np.linspace(0,1,round(n_points))*fs/2
    new_psd = np.interp(np.log10(new_fpsd), np.log10(fpsd), np.log10(psd), left=-30, right=-30)
    new_psd = 10**new_psd
    new_psd[new_psd<=1e-30]= 0
    new_psd[0] = 0
    return new_fpsd, new_psd
    
def get_stat_mom(var, probability, order):
    """
    Compute statistical moment of a given order from a PDF.

    Parameters:
        var (np.ndarray): Grid of variable values.
        probability (np.ndarray): PDF values corresponding to `var`.
        order (int): Moment order.

    Returns:
        float: Statistical moment of the given order.
    """
    return np.trapezoid(probability*var**order,var)

def alpha_spec_index(fpsd, psd, order, deriv_order):
    """
    Compute the spectral index (alpha) of a process.

    Parameters:
        fpsd (np.ndarray): Frequency vector.
        psd (np.ndarray): Power spectral density.
        order (int): Order of the moment.
        deriv_order (int): Derivative order.

    Returns:
        float: Spectral index.
    """
    m2n_x = get_stat_mom(fpsd, psd, 2*deriv_order+order)
    m2n = get_stat_mom(fpsd, psd, 2*deriv_order)
    m2n_2x = get_stat_mom(fpsd, psd, 2*(deriv_order+order))
    return m2n_x/np.sqrt(m2n*m2n_2x)

def get_psd_impulse_response(fpsd,psd, N, fs = None):
    """
    Compute an impulse response from a target PSD.

    Parameters:
        fpsd (np.ndarray): Frequency vector.
        psd (np.ndarray): PSD values.
        N (int): Number of samples in the impulse response.

    Returns:
        np.ndarray: Time-domain impulse response.
    """
    if fs is None: 
        fs = fpsd[-1] * 2
    N_os = (N // 2 + 1)
    _, psd_interp = lin_interp_psd(fpsd, psd,fs, N_os)
    T = N/(fs)
    freq_filter = psd_interp*T*fpsd[1]
    h_t = np.fft.irfft((np.sqrt(freq_filter)), n = N)*fs
    h_t = np.roll(h_t, N//2) # circular shift to make the impulse centred
    return h_t

def get_stat_history(signal:np.ndarray, winsize:int, olap:int = 0, idx_type = 'rms'):
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
    idx_types = {'mean':np.mean,
                 'rms':np.std,
                 'var':np.var,
                 'kurtosis':sp.stats.kurtosis}
    
    method_existance_check(idx_type, idx_types)
    
    winsize = int(np.round(winsize))
    olap = int(np.round(olap))
    step = winsize-olap

    idx_history = np.zeros((len(signal) - winsize + 1)//(step)+1)
    
    print(f'Calculating {idx_type} evolution')
    k = 0
    for i in tqdm(range(0, len(signal) - winsize + 1, step)):
        index =  idx_types[idx_type](signal[i : i + winsize])
        idx_history[k] = index
        k += 1
    if 'kurtosis' in idx_type:
        idx_history = idx_history+3
    return idx_history

def get_nnst_index(signal, nperseg = 100, noverlap = 0, *args, **kwargs):
    """
    Non-stationarity index identification using modified run-test.
    Standard deviation of entire signal is compared to standard deviation of segmented signal,
    and the number of variations (i.e., runs) is compared to expected value of variations to obtain
    the non-stationarity index.

    For the complete documentation please refer to: https://github.com/LolloCappo/pyNNST

    Parameters
    ---------- 
    signal: array-like
        One dimensional time history
    nperseg : int
        Length of each segment
    
    noverlap: int, optional
        Number of points to overlap between segments. If None,
        "noverlap = 0". Defaults to None.

    confidence: int, optional
        Confidence interval [90-95-98-99] %. If None, 
        "confidence = 95". Defaults to None.    
    
    *args, **kwargs: optional
        Additional arguments for the method
    
    Returns
    -------
    test_results: dict
        Results of the non-stationary test    
    
    Source
    ------
    L. Capponi, M. Česnik, J. Slavič, F. Cianetti, e M. Boltežar, «Non-stationarity index in vibration fatigue: Theoretical and experimental research», Int. J. Fatigue, vol. 104, pp. 221–230, nov. 2017, doi: 10.1016/j.ijfatigue.2017.07.020.
    """
    test = pyNNST.nnst(signal, nperseg = nperseg, noverlap=noverlap, *args, **kwargs)
    test.idns() # perform index assessment
    test_results = {
        'outcome': test.get_outcome(),
        'test': test.get_index(),
        'winsize': test.nperseg,
        'olap': test.noverlap,
        'confidence': test.confidence,
    }
    return test_results

def get_adf_index(signal, maxlag=None, regression='c', autolag='AIC', *args, **kwargs):
    """
    Augmented Dickey-Fuller unit root test from the statsmodels package.

    The Augmented Dickey-Fuller test can be used to test for a unit root in a univariate process in the presence of serial correlation.

    For the complete documentation please refer to: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html#statsmodels.tsa.stattools.adfuller

    Parameters
    ----------
    signal : array_like, 1d
        The data series to test.
    maxlag : {None, int}
        Maximum lag which is included in test, default value of
        12*(nobs/100)^{1/4} is used when ``None``.
    regression : {"c","ct","ctt","n"}
        Constant and trend order to include in regression.

        * "c" : constant only (default).
        * "ct" : constant and trend.
        * "ctt" : constant, and linear and quadratic trend.
        * "n" : no constant, no trend.

    autolag : {"AIC", "BIC", "t-stat", None}
        Method to use when automatically determining the lag length among the
        values 0, 1, ..., maxlag.

        * If "AIC" (default) or "BIC", then the number of lags is chosen to minimize the corresponding information criterion.
        * "t-stat" based choice of maxlag.  Starts with maxlag and drops a lag until the t-statistic on the last lag length is significant using a 5%-sized test.
        * If None, then the number of included lags is set to maxlag.

    *args, **kwargs : optional
        Additional arguments for the method

    Returns
    -------
    test_results : dict
        Dictionary containing the non-stationarity test results:
        'outcome', 'test', 'p-value', 'lag', 'crit_values'

    Source
    ------
    Dickey, D. A., and W. A. Fuller. "Distribution of the Estimators for Autoregressive Time Series with a Unit Root." Journal of the American Statistical Association. Vol. 74, 1979, pp. 427–431.
    """
    test = adfuller(signal, maxlag, regression, autolag, *args, **kwargs)
    if test[1] < 0.05: # p-value
        outcome = 'Stationary'
    else:
        outcome = 'Non-stationary'
    test_results = {
        'outcome': outcome,
        'test': test[0],
        'p-value': test[1],
        'lag': test[2],
        'crit_values': test[4]
        }    
    return test_results

def get_kpss_index(signal, regression = 'c', nlags = 'auto', *args, **kwargs):
    """
    Kwiatkowski-Phillips-Schmidt-Shin test for stationarity.

    Computes the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for the null
    hypothesis that x is level or trend stationary.

    For the complete documentation please refer to: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html#statsmodels.tsa.stattools.kpss

    Parameters
    ----------
    signal : array_like, 1d
        The data series to test.
    regression : str{"c", "ct"}
        The null hypothesis for the KPSS test.

        * "c" : The data is stationary around a constant (default).
        * "ct" : The data is stationary around a trend.
        
    nlags : {str, int}, optional
        Indicates the number of lags to be used. If "auto" (default), lags
        is calculated using the data-dependent method of Hobijn et al. (1998).
        See also Andrews (1991), Newey & West (1994), and Schwert (1989). If
        set to "legacy",  uses int(12 * (n / 100)**(1 / 4)) , as outlined in
        Schwert (1989).
    *args, **kwargs : optional
        Additional arguments for the method

    Returns
    -------
    test_results : dict
        Dictionary containing the non-stationarity test results:
        'outcome', 'test', 'p-value', 'lag', 'crit_values'
    
    Source
    ------
    Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., Shin, Y. (1992). "Testing the null hypothesis of stationarity against the alternative of a unit root". Journal of Econometrics. 54 (1–3): 159–178. doi:10.1016/0304-4076(92)90104-Y.
    """
    test = kpss(signal, regression=regression, nlags=nlags, *args, **kwargs)
    if test[1] < 0.05: # p-value
        outcome = 'Non-stationary'
    else:
        outcome = 'Stationary'
    
    test_results = {
        'outcome': outcome,
        'test': test[0],
        'p-value': test[1],
        'lag': test[2],
        'crit_values': test[3]
        }
    return test_results

def print_nonstat_results(ns_test, indent=0):
    for key, value in ns_test.items():
        print(' ' * indent + str(key) + ':\t', end=' ')
        if isinstance(value, dict):
            print()
            print_nonstat_results(value, indent + 4)
        else:
            print(value)
    pass

def get_banded_spectral_kurtosis(
    signal: np.ndarray, 
    dt:float, 
    n_bins:int = 10, 
    fl : int = 0, 
    fu : int = 0
    ):# * GCurti's theory
    """
    Compute spectral kurtosis in frequency bands.

    Parameters:
        signal (np.ndarray): Input time history.
        dt (float): Sampling interval.
        n_bins (int): Number of frequency bands.
        fl (float): Lower frequency limit.
        fu (float): Upper frequency limit (default is Nyquist).

    Returns:
        tuple: (center frequencies, spectral kurtosis values)
    """
    f_ny = 1/2/dt
    if fu <= fl: fu = f_ny
    if not isinstance(n_bins, int): 
        n_bins = round(n_bins)
    Nos = np.ceil(len(signal)/2)
    Xts_full_woShift = dt*np.fft.fft(signal)
    Xos_full = Xts_full_woShift[0:int(Nos)]
    fper = np.linspace(0, f_ny, len(Xos_full))
    N = len(Xts_full_woShift)
    binsize = (fu-fl) / n_bins
    fvec = np.arange(n_bins) * binsize + fl + binsize/ 2
    SK = np.zeros(n_bins)
    print('Evaluating kurtosis on frequency bands:')
    for i in tqdm(np.arange(n_bins)):
        Xos = np.zeros(len(Xos_full), dtype=np.complex128)
        Xos[(fper>=(fl+binsize*i)) & (fper<=(fl+binsize*(i+1)))] = Xos_full[(fper>=(fl+binsize*i)) & (fper<=(fl+binsize*(i+1)))]
        x = np.fft.irfft(Xos,n = N)/dt
        SK[i] = sp.stats.kurtosis(x)+3    
    return fvec, SK

def get_welch_spectral_kurtosis(x:np.ndarray, Nfft:int, noverlap:int = None):
    """
    Compute Welch-based spectral kurtosis.

    Parameters:
        x (np.ndarray): Input signal.
        Nfft (int): FFT size.
        noverlap (int): Number of overlapping samples.

    Returns:
        tuple: (normalized frequency vector, spectral kurtosis)
    
    Source:
    J. Antoni, «The spectral kurtosis: a useful tool for characterising non-stationary signals», Mech. Syst. Signal Process., vol. 20, fasc. 2, pp. 282–307, feb. 2006, doi: 10.1016/j.ymssp.2004.09.001.
    """
    if noverlap is None:
        noverlap = Nfft//2
    
    # Convert scalar window input to actual window
    Window = sp.signal.windows.hann(Nfft//2)
    
    Window = np.asarray(Window).flatten()
    Window = Window / np.linalg.norm(Window)  # normalize window

    x = np.asarray(x)
    n = len(x)
    nwind = len(Window)

    if nwind <= noverlap:
        raise ValueError('Window length must be > noverlap')
    if Nfft < nwind:
        raise ValueError('Window length must be <= Nfft')

    step = nwind - noverlap
    k = (n - noverlap) // step  # number of windows

    M4 = np.zeros(Nfft)
    M2 = np.zeros(Nfft)

    for i in range(k):
        start = i * step
        xw = Window * x[start:start + nwind]
        Xw = np.fft.fft(xw, Nfft)
        absX = np.abs(Xw)
        M4 += absX**4
        M2 += absX**2

    M4 /= k
    M2 /= k
    SK = M4 / (M2**2) - 2
    
    # Bias correction
    W = np.abs(np.fft.fft(Window**2, Nfft))**2
    Wb = np.array([W[(2*i) % Nfft] / W[0] for i in range(Nfft)])
    SK -= Wb

    f_norm = np.arange(Nfft) / Nfft

    return f_norm[:Nfft//2], SK[:Nfft//2]

def get_stationary_gaussian(fpsd:np.ndarray, psd:np.ndarray, T:float, fs:float = None, seed = None, interp:str = 'lin'):
    """
    Generate a stationary Gaussian signal with a target PSD.

    Parameters:
        fpsd (np.ndarray): Frequency vector.
        psd (np.ndarray): Power spectral density values.
        T (float): Duration [s].
        seed (int, optional): Random seed for reproducibility.

    Returns:
        tuple: (time vector, generated signal)
    
    Source:
    D. E. Newland, An introduction to random vibrations, spectral & wavelet analysis, 3. ed., Unabridged republ. Mineola, NY: Dover Publ, 2005
    """    
    fpsd = np.array(fpsd)    
    psd = np.array(psd)

    psd[np.isnan(psd)] = 0

    if fs is None: 
        fs = fpsd[-1] * 2
    
    N = int(fs * T)

    N_os = N // 2 + 1

    # Ensure zero-frequency component exists
    # if fpsd[0] != 0:
    #     fpsd = np.concatenate(([0], fpsd))
    #     fpsd = np.concatenate(([0], psd))

    # fvec = np.linspace(0, fs // 2, N_os)
    mue2 = np.trapezoid(psd, fpsd)

    # Interpolate PSD onto frequency vector
    interpolators ={
        'lin':lin_interp_psd,
        'log':log_interp_psd
    }
    method_existance_check(interp, interpolators)
    # per = np.interp(fvec, fpsd, psd)
    fper, per = interpolators[interp](fpsd, psd, N_os, fs)
    per = mue2 / (np.sum(per / T)) * per  # Equivalent to dfpsd * inp['T'] * per

    # Generate random phase and magnitude
    Xabs = np.sqrt(per * T / 2)
    if seed is not None:
        np.random.seed(seed)
    Xos = Xabs * np.exp(1j * np.random.uniform(-np.pi, np.pi, len(Xabs)))
    np.random.seed(None)  # Reset random seed

    # Compute time-domain signal
    signal = np.fft.irfft(Xos * fs, n=N)
    t = np.arange(0, N) / fs

    return t, signal  # Return time-domain signal

def gaussian_bell(grid, var, mean = 0):
    """
    Evaluate a Gaussian distribution over a grid.

    Parameters:
        grid (np.ndarray): Input grid.
        var (float): Variance.
        mean (float, optional): Mean. Default is 0.

    Returns:
        np.ndarray: Evaluated Gaussian PDF values.
        
    Source: 
    Squires, G. L. (2001-08-30). Practical Physics (4 ed.). Cambridge University Press. doi:10.1017/cbo9781139164498. ISBN 978-0-521-77940-1.
    """
    return 1/ np.sqrt(var*2*np.pi) * np.exp(-(grid-mean)**2 / (2 * var))
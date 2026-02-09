import numpy as np
import scipy as sp
from tqdm import tqdm


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
    from pyNNST import nnst
    test = nnst(signal, nperseg = nperseg, noverlap=noverlap, *args, **kwargs)
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
    from statsmodels.tsa.stattools import adfuller
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
    from statsmodels.tsa.stattools import kpss
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
    fvec = (np.arange(n_bins) * binsize + fl + binsize/ 2)*dt
    SK = np.zeros(n_bins)
    print('Evaluating kurtosis on frequency bands:')
    for i in tqdm(np.arange(n_bins)):
        Xos = np.zeros(len(Xos_full), dtype=np.complex128)
        Xos[(fper>=(fl+binsize*i)) & (fper<=(fl+binsize*(i+1)))] = Xos_full[(fper>=(fl+binsize*i)) & (fper<=(fl+binsize*(i+1)))]
        x = np.fft.irfft(Xos,n = N)/dt
        SK[i] = sp.stats.kurtosis(x)    
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

def fast_kurtogram(x, fs, nlevel=7):
    '''
    Fast Kurtogram computation
    
    Parameters:
        x       : input signal (1D array)
        nlevel  : number of decomposition levels (Maximum number of decomposition levels is log2(length(x)), but it is recommended to stay by a factor 1/8 below this.)
        fs      : sampling frequency of signal x (default is Fs = 1)
        plot    : boolean to enable/disable plotting of results (default is False)

    Output:
        Kwav        : 2D ndarray of shape (2 * nlevel, 3 * 2**nlevel)
                    Kurtosis map (Fast Kurtogram). Each entry Kwav[i, j]
                    is the envelope kurtosis of the sub-band corresponding
                    to level Level_w[i] and center frequency freq_w[j].

        Level_w     : 1D ndarray of length 2 * nlevel
                    Effective decomposition levels associated with each row
                    of Kwav. It contains both binary and ternary levels,
                    ordered as in the original MATLAB implementation
                    (top row = level 0).

        freq_w      : 1D ndarray of length 3 * 2**nlevel
                    Center frequencies [Hz] of each sub-band along the
                    horizontal axis (columns) of Kwav.

        fc          : float
                    Optimal center frequency [Hz] corresponding to the
                    maximum kurtosis in Kwav (i.e. frequency of the most
                    impulsive band).

        bandwidth   : float
                    Bandwidth [Hz] of the optimal sub-band associated with fc.

        max_kurt    : float
                    Maximum kurtosis value found in Kwav.

        level_max   : float
                    Effective decomposition level (in Level_w) at which the
                    maximum kurtosis max_kurt occurs.

        c           : 1D ndarray
                    Signal filtered in the optimal sub-band (center frequency fc
                    and bandwidth bandwidth). This is the band-limited signal
                    typically used for further analysis (e.g., envelope analysis
                    and envelope spectrum for fault detection).
    References:
        J. Antoni, Fast Computation of the Kurtogram for the Detection of Transient Faults, Mechanical Systems and Signal Processing, Volume 21, Issue 1, 2007, pp.108-124.
    
    '''
    
    firwin = sp.signal.firwin
    lfilter = sp.signal.lfilter
    
    # INTERNAL FUNCTIONS
    def _kurt(this_x,opt):
    
        eps = 2.2204e-16

        if opt.lower() == 'kurt2':
            if np.all(this_x == 0):
                K = 0
                return K
            this_x -= np.mean(this_x)
            
            E = np.mean(np.abs(this_x)**2)
            
            if E < eps:
                K = 0
                return K
            K = np.mean(np.abs(this_x)**4) / E**2
            
            if np.all(np.isreal(this_x)):
                K -= 3
            else:
                K -= 2
        elif opt.lower() == 'kurt1':
            if np.all(this_x == 0):
                K = 0
                return K
            this_x = this_x - np.mean(this_x)
            E = np.mean(np.abs(this_x))
            
            if E < eps:
                K = 0
                return K
            
            K = np.mean(np.abs(this_x)**2) / E**2
            
            if np.all(np.isreal(this_x)):
                K -= 1.57
            else:
                K -= 1.27        
                
        
        return K


    def _K_wpQ(x,h,g,h1,h2,h3,nlevel,opt,level=None):
        '''
        Computes the kurtosis K of the complete "binary-ternary" wavelet packet transform w of signal x, 
        up to nlevel, using the lowpass and highpass filters h and g, respectively. 
        The values in K are sorted according to the frequency decomposition.
        '''
        
        if level == None:
            level = nlevel
            
        x = x.flatten()
        L = np.floor(np.log2(x.size))
        x = np.atleast_2d(x).T
        
        KD,KQ = _K_wpQ_local(x,h,g,h1,h2,h3,nlevel,opt,level)
        
        K = np.zeros((2 * nlevel,3 * 2**nlevel))
        
        K[0,:] = KD[0,:]
        
        for i in np.arange(1,nlevel):
            K[2*i-1,:] = KD[i,:]
            K[2*i,:] = KQ[i-1,:]
        
        K[2*nlevel-1,:] = KD[nlevel,:]
        

        return K
        
    def _K_wpQ_local(x,h,g,h1,h2,h3,nlevel,opt,level):
        
        
        
        a,d = _DBFB(x,h,g)
        
        N = np.amax(a.shape)
        
        d = d * (-1)**(np.atleast_2d(np.arange(1,N+1)).T)

        Lh = np.amax(h.shape)
        Lg = np.amax(g.shape)
            
        K1 = _kurt(a[Lh-1:],opt)
        K2 = _kurt(d[Lg-1:],opt)
        
        if level > 2:
            a1,a2,a3 = _TBFB(a,h1,h2,h3)
            d1,d2,d3 = _TBFB(d,h1,h2,h3)
            
            Ka1 = _kurt(a1[Lh-1:],opt)
            Ka2 = _kurt(a2[Lh-1:],opt)
            Ka3 = _kurt(a3[Lh-1:],opt)
            Kd1 = _kurt(d1[Lh-1:],opt)
            Kd2 = _kurt(d2[Lh-1:],opt)
            Kd3 = _kurt(d3[Lh-1:],opt)
            
        else:
            Ka1 = 0
            Ka2 = 0
            Ka3 = 0
            Kd1 = 0
            Kd2 = 0
            Kd3 = 0
        
        if level == 1:
            K = np.concatenate((K1 * np.ones(3),K2 * np.ones(3)))
    #         print(K.shape)
            KQ = np.array([Ka1,Ka2,Ka3,Kd1,Kd2,Kd3])

        if level > 1:
            
            Ka,KaQ = _K_wpQ_local(a,h,g,h1,h2,h3,nlevel,opt,level-1)
            Kd,KdQ = _K_wpQ_local(d,h,g,h1,h2,h3,nlevel,opt,level-1)
            

            K1 *= np.ones(np.amax(Ka.shape))
            K2 *= np.ones(np.amax(Kd.shape))
            
            
            K = np.vstack((np.concatenate([K1,K2]),
                        np.hstack((Ka,Kd))))
            
            
            Long = int(2/6 * np.amax(KaQ.shape))
            Ka1 *= np.ones(Long)
            Ka2 *= np.ones(Long)
            Ka3 *= np.ones(Long)
            Kd1 *= np.ones(Long)
            Kd2 *= np.ones(Long)
            Kd3 *= np.ones(Long)
            
            KQ = np.vstack((np.concatenate([Ka1,Ka2,Ka3,Kd1,Kd2,Kd3]),
                            np.hstack((KaQ,KdQ))))
            
            

        if level == nlevel:
            
            K1 = _kurt(x,opt)
            
            K = np.vstack((K1 * np.ones(np.amax(K.shape)),K))
            
            a1,a2,a3 = _TBFB(x,h1,h2,h3)
            
            Ka1 = _kurt(a1[Lh-1:],opt)
            Ka2 = _kurt(a2[Lh-1:],opt)
            Ka3 = _kurt(a3[Lh-1:],opt)
            
            Long = int(1/3 * np.amax(KQ.shape))
            
            Ka1 *= np.ones(Long)
            Ka2 *= np.ones(Long)
            Ka3 *= np.ones(Long)

            
            KQ = np.vstack((np.concatenate([Ka1,Ka2,Ka3]),
                            KQ[:-2,:]))

        return K,KQ
        
            
        
    def _TBFB(x,h1,h2,h3):
        
        N = x.flatten().size
        
        a1 = lfilter(h1,1,x.flatten())
        a1 = a1[2:N:3]
        a1 = np.atleast_2d(a1).T
        
        a2 = lfilter(h2,1,x.flatten())
        a2 = a2[2:N:3]
        a2 = np.atleast_2d(a2).T
        
        a3 = lfilter(h3,1,x.flatten())
        a3 = a3[2:N:3]
        a3 = np.atleast_2d(a3).T
        
        return a1,a2,a3
        
    def _DBFB(x,h,g):
        
        N = x.flatten().size
        
        a = lfilter(h,1,x.flatten())
        a = a[1:N:2]
        a = np.atleast_2d(a).T
        
        d = lfilter(g,1,x.flatten())
        
        d = d[1:N:2]
        
        d = np.atleast_2d(d).T
        
        return a,d

    def binary(i,k):
        
        k = int(k)
        
        if i > 2**k:
            raise ValueError('i must be such that i < 2^k')
        
        a = np.zeros(k)
        
        temp = i
        
        for l in np.arange(k)[::-1]:
            a[-(l+1)] = np.fix(temp / 2**l)
            temp -= a[-(l+1)] * 2 ** l
        
        return a


    def find_wav_kurt(x,h,g,h1,h2,h3,Sc,Fr,opt,Fs):
        level = np.fix(Sc) + (np.remainder(Sc,1)>=0.5) * (np.log2(3)-1)
        
        Bw = 2**(-level - 1)
        freq_w = np.arange(2**level) / (2**(level+1)) + Bw/2
        J = np.argmin(np.abs(freq_w - Fr))
        fc = freq_w[J]
        i = np.round((fc/Bw - 1/2))
        
        
        
        if np.remainder(level,1) == 0:
            
            acoeff = binary(i,level)
            bcoeff = np.array([])
            temp_level = level
        
        else:
            
            i2 = np.fix(i/3)
            temp_level = np.fix(level) - 1
            acoeff = binary(i2,temp_level)
            bcoeff = i - i2 * 3
        
        acoeff = acoeff[::-1]

        c = K_wpQ_filt(x,h,g,h1,h2,h3,acoeff,bcoeff,temp_level)
        
        kx = _kurt(c,opt)
        
        sig = np.median(np.abs(c)) / np.sqrt(np.pi / 2)
        
        threshold = sig * np.sqrt((-2*1**2) * np.log(1 - 0.999))
        
        return c, Bw, fc, i
        
    def K_wpQ_filt(x,h,g,h1,h2,h3,acoeff,bcoeff,level=None):
        
        nlevel = acoeff.size
        
        L = np.floor(np.log2(np.amax(x.shape)))
        
        if level == None:
            if nlevel >= L:
                raise ValueError('nlevel must be smaller')
            
            level = nlevel
        
        x = np.atleast_2d(x.flatten()).T

        if nlevel == 0:
            if bcoeff.size == 0:
                c = x
            else:
                c1, c2, c3 = _TBFB(x,h1,h2,h3)
                
                if bcoeff == 0:
                    c = c1[h1.size - 1:]
                elif bcoeff == 1:
                    c = c2[h2.size - 1:]
                elif bcoeff == 2:
                    c = c3[h3.size - 1:]
            
        else:
            
            c = K_wpQ_filt_local(x,h,g,h1,h2,h3,acoeff,bcoeff,level)

        
        return c

    def K_wpQ_filt_local(x,h,g,h1,h2,h3,acoeff,bcoeff,level):
        
        a,d = _DBFB(x,h,g)
        
        N = a.size
        
        level = int(level)
        
        d = d*np.array([(-1)**(np.arange(1,N+1))]).T
        
        if level == 1:
            if bcoeff.size == 0:
                if acoeff[level-1] == 0:
                    c = a[h.size-1:]
                else:
                    c = d[g.size-1:]
            else:
                if acoeff[level-1] == 0:
                    c1,c2,c3 = _TBFB(a,h1,h2,h3)
                else:
                    c1,c2,c3 = _TBFB(d,h1,h2,h3)
                
                if bcoeff == 0:
                    c = c1[h1.size - 1:]
                elif bcoeff == 1:
                    c = c2[h2.size - 1:]
                elif bcoeff == 2:
                    c = c3[h3.size - 1:]
        
        if level > 1:
            if acoeff[level-1] == 0:
                c = K_wpQ_filt_local(a,h,g,h1,h2,h3,acoeff,bcoeff,level-1)
            else:
                c = K_wpQ_filt_local(d,h,g,h1,h2,h3,acoeff,bcoeff,level-1)
                
        return c
        
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # % CHECK INPUT VALUES
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    N = x.flatten().size
    N2 = np.log2(N) - 7

    if nlevel > N2:
        raise ValueError('Please enter a smaller number of decomposition levels')

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # % FAST COMPUTATION OF THE KURTOGRAM (by means of wavelet packets or STFT)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    x = x - np.mean(x)

    # Binary (2-band) analytic filters / Analytic generating filters :
    Nfir = 16
    fc0 = 0.4
    # a short filter is just good enough!

    h = firwin(Nfir+1, fc0) * np.exp(2j * np.pi * np.arange(Nfir+1) * 0.125)
    n = np.arange(2, Nfir+2) 

    # High-pass filter g (Antoni formulation)
    g = h[(1-n) % Nfir] * (-1.)**(1-n) # why not -1j? and why 1-n?

    # Ternary (3-band) refinement filters
    Nfir3 = int(np.fix(3/2 * Nfir))
# 
    h1 = firwin(Nfir3+1, 2/3 * fc0) * np.exp(2j * np.pi * np.arange(0, Nfir3+1) * (0.25/3))
    h2 = h1 * np.exp(2j * np.pi * np.arange(0, Nfir3+1) / 6)
    h3 = h1 * np.exp(2j * np.pi * np.arange(0, Nfir3+1) / 3)

    # Full binary+ternary wavelet packet kurtosis
    Kwav = _K_wpQ(x, h, g, h1, h2, h3, nlevel, 'kurt2')     # kurtosis of the complex envelope
    Kwav = np.clip(Kwav, 0, np.inf)                         # keep positive values only!


    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # % GRAPHICAL DISPLAY OF RESULTS
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Level axis (same as MATLAB)
    Level_w = np.arange(1, nlevel+1, dtype=float)
    Level_w = np.vstack((Level_w, Level_w + np.log2(3)-1)).reshape(-1, order='F')
    Level_w = np.insert(Level_w, 0, 0.0)[:2*nlevel]

    # Frequency axis
    freq_w = fs * (np.arange(3*2**nlevel)/(3*2**(nlevel+1)) + 1/(3*2**(2+nlevel)))

    # Find max kurtosis (row-wise max, then global)
    row_max = Kwav[np.arange(Kwav.shape[0]), np.argmax(Kwav, axis=1)]
    max_level_index = np.argmax(row_max)
    max_kurt = row_max[max_level_index]
    level_max = Level_w[max_level_index]

    # Bandwidth and center frequency from (level_max, J)
    bandwidth = fs * 2**(-(level_max + 1))

    J = np.argmax(Kwav[max_level_index, :])

    fi = J / (3 * 2**(nlevel + 1))               # normalized
    fi = fi + 2**(-2 - level_max)                # normalized
    fc = fs * fi                                 # Hz
    
        # Extract optimal band / envelope
    c, _, _, _ = find_wav_kurt(x, h, g, h1, h2, h3, level_max, fi, 'kurt2', fs)
    
    
    return Kwav, Level_w, freq_w, fc, bandwidth ,max_kurt,level_max,c

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

def perfect_passband_filter(signal, fl_n, fu_n):
    """_summary_

    Args:
        signal (_type_): timehistory
        fl_n (_type_): low frequency given normalized against the sampling frequency
        fu_n (_type_): low frequency given normalized against the sampling frequency
    """
    fft_os = np.fft.rfft(signal)
    f_fft_os = np.fft.rfftfreq(signal.size)
    mask = (np.abs(f_fft_os)>fl_n) & (np.abs(f_fft_os)<fu_n)
    filtered_fft = np.zeros(len(fft_os), dtype=complex)
    filtered_fft[mask] = fft_os[mask]
    filtered_signal = np.fft.irfft(filtered_fft)
    return filtered_signal
    
    
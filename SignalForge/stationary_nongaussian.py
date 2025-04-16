import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from utils import *
from single_chan_signal import SingleChanSignal


'''
supporting functions / 'static methods' (no need of self object)
- get_winterstein:             calculate non-gaussian signal using the Winterstein method
- get_cubic_polinomial:        calculate non-gaussian signal using the cubic polymomial method
- get_zheng:                   calculate non-gaussian signal using the hyperbolic method
- get_sarkani:                 calculate non-gaussian signal using the Sarkani method
- get_zmnl:                    calculate non-gaussian signal using the improved non-linear method
- get_steinwolf:               calculate non-gaussian signal using the Steinwolf method
- get_smallwood:               calculate non-gaussian signal using the Smallwood shotnoise method
- get_vanbaren:                calculate non-gaussian signal using the Vanbaren method
- get_arma:                    calculate non-gaussian signal using the ARMA method

'''


def _get_signal_statistics(signal: np.ndarray) -> tuple:
    """
    Internal function to compute basic statistics and theoretical Gaussian PDF for a signal.

    Parameters
    ----------
        signal (np.ndarray): Input signal array.

    Returns
    ----------
        tuple: A tuple containing:
            - rms (float): Root mean square (standard deviation) of the signal.
            - grid (np.ndarray): Grid over which the PDF is evaluated.
            - pdf_x (np.ndarray): Bestfit Gaussian PDF values evaluated over the grid.
    """
    # Compute the root mean square (standard deviation) of the signal
    rms = np.std(signal)

    # Generate a grid of values spanning ±10 times the RMS
    grid = np.linspace(-10 * rms, 10 * rms, 5000)

    # Evaluate a Gaussian PDF with variance equal to rms squared over the grid
    pdf_x = gaussian_bell(grid, var=rms**2)

    return rms, grid, pdf_x

def get_winterstein(
    signal: np.ndarray,
    input_skewness: float,
    input_kurtosis: float,
    params: dict = {'order': 3}
) -> tuple:
    """
    Applies the Winterstein method to generate a non-Gaussian signal
    with specified skewness and kurtosis from a Gaussian input.

    Parameters
    ----------
        signal (np.ndarray): 
            Input signal.
        input_skewness (float): 
            Desired skewness of the output signal.
        input_kurtosis (float):     
            Desired kurtosis of the output signal.
        params (dict): 
            Optional dictionary of parameters. Currently supports:
            - 'order': 2 or 3, for second or third order Winterstein transformation.

    Returns
    ---------
        tuple: A tuple containing:
            - ng_signal (np.ndarray): Transformed non-Gaussian signal.
            - opt_params (None): Placeholder for compatibility (no optimization done here).
            
    Source
    ------
    S. R. Winerstein, «Nonlinear Vibration Models for Extremes and Fatigue», J. Eng. Mech., vol. 114, fasc. 10, pp. 1772-1790, ott. 1988, doi: 10.1061/(ASCE)0733-9399(1988)114:10(1772)
    """
    
    if input_kurtosis > 12:
        print(
            'Kurtosis requested may be too high for correct representation via '
            'Winterstein method. Consider using alternative methods.'
        )

    K = input_kurtosis
    S = input_skewness

    # Normalize input signal to zero-mean, unit-variance
    input_rms = sp.stats.tstd(signal)
    x = signal / input_rms

    # Choose transformation order
    if params.get('order', 3) == 2:
        print('Estimating Winterstein second order non-linear transform...')
        h_3 = S / (4 + 2 * np.sqrt(1 + 1.5 * (K - 3)))
        h_4 = (np.sqrt(1 + 1.5 * (K - 3)) - 1) / 18
    else:
        print('Estimating Winterstein third order non-linear transform...')
        h_3 = (S / 6) * (1 - 0.015 * np.abs(S) + 0.3 * S**2) / (1 + 0.2 * (K - 3))
        h_4 = ((1 + 1.12 * (K - 3))**(1/3) - 1) / 10
        h_4 *= (1 - 1.43 * S**2 / (K - 3))**(1 - 0.1 * K**0.8)

    h_0 = 1 / np.sqrt(1 + 2 * h_3**2 + 6 * h_4**2)

    # Apply the appropriate transformation
    if K > 3:
        ng_signal = h_0 * (x + h_3 * (x**2 - 1) + h_4 * (x**3 - 3 * x))
    else:
        c_0 = -h_3 / (3 * h_4)
        c_1 = (1 / (2 * h_4)) * (h_3 / (3 * h_4) + x / h_0) - (h_3 / (3 * h_4))**3
        c_2 = 1 / (3 * h_4) - (h_3**2 / (9 * h_4**2)) - 1
        ng_signal = (
            c_0
            + (c_1 + np.sqrt(c_1**2 + c_2**3))**(1 / 3)
            + (c_1 - np.sqrt(c_1**2 + c_2**3))**(1 / 3)
        )

    # Rescale to original RMS
    output_rms = sp.stats.tstd(ng_signal)
    ng_signal *= input_rms / output_rms

    opt_params = None
    return ng_signal, opt_params

def get_cubic_polinomial(signal:np.ndarray, input_kurtosis:float, input_skewness:float = None): 
    """
    Applies a cubic polynomial transformation to a signal to match specified skewness and kurtosis.

    Parameters
    -------
        signal (np.ndarray): 
            Input Gaussian signal.
        input_kurtosis (float): 
            Target kurtosis.
        input_skewness (float): 
            Target skewness.

    Returns
    ------
        Tuple[np.ndarray, dict]: 
            Transformed signal and dictionary of optimal parameters.
    Source
    ----------
    S. R. Winterstein, C. H. Lange, e S. Kumar, «Fitting: Subroutine to fit four-moment probability distributions to data», SAND--94-3039, 10123319, gen. 1995. doi: 10.2172/10123319.
    """
    
    def cubic_transform(x, a):
        """Cubic polynomial transformation."""
        x_ng = a[0]*x+a[1]*x**2+a[2]*x**3 
        return x_ng
    
    def nonlinear_constraint(a):
        """Nonlinear constraint to ensure a physically feasible cubic polynomial."""
        a1, a2, a3 = a
        return np.sqrt(3*a1*a3) - np.abs(a2)  # Must be >= 0
    
    ## * OBJECTIVE FUNCTION WITH TIMEHISTORY
    # def cubic_obj_fun(var):
    #     nonlocal x, input_rms, input_kurtosis
    #     z = cubic_transform(x, var)
    #     obj_val = abs(np.abs(sp.stats.kurtosis(z) + 3 - input_kurtosis)  + np.abs(sp.stats.skew(z) - input_skewness))
    #     return obj_val
    
    def cubic_obj_fun(var):
        """
        Objective function to minimize the absolute error in kurtosis and skewness after transformation.
        """
        nonlocal x, pdf_x, input_rms, input_kurtosis, input_skewness
        z = cubic_transform(grid, var)
        
        ## * FOR EQUIVALENCE TO TIMESERIES TEST
        # zt = cubic_transform(x, var)
        # time_kurt = sp.stats.kurtosis(zt) + 3

        transformed_kurtosis = np.trapz(pdf_x*z**4,grid)/np.trapz(pdf_x*z**2,grid)**2
        transformed_skewness = np.trapz(pdf_x*z**3,grid)/np.trapz(pdf_x*z**2,grid)**(3/2)
        obj_val = np.abs(transformed_kurtosis - input_kurtosis) + np.abs(transformed_skewness - input_skewness)
        return obj_val
    
    x = signal
    input_rms, grid, pdf_x = _get_signal_statistics(x)
    constraint = {'type': 'ineq', 'fun': nonlinear_constraint}
    optim_res = sp.optimize.minimize(
        cubic_obj_fun, 
        x0=[0.1, 0, 0.1], 
        bounds=[(0, 100), (-100, 100), (0, 100)], 
        constraints=[constraint]
        )
    ng_signal = cubic_transform(x, optim_res.x)
    ng_signal = ng_signal*(input_rms/np.std(ng_signal))
    opt_params = {'a': optim_res.x}
    return ng_signal, opt_params

def get_zheng(signal:np.ndarray, input_kurtosis:float, input_skewness:float = None):
    """
    Applies Zheng's transformation to achieve a desired kurtosis and skewness.

    Parameters
    ----------
        signal (np.ndarray): 
            Input Gaussian signal.
        input_kurtosis (float): 
            Target kurtosis.
        input_skewness (float, optional): 
            Target skewness. If None, assumes symmetry.

    Returns
    -------
        Tuple[np.ndarray, dict]: 
            Transformed signal and optimal parameters.
    Source
    ------
    R. Zheng, H. Chen, e X. He, «Control Method for Multiple-Input Multiple-Output Non-Gaussian Random Vibration Test: MIMO Non-Gaussian Random Vibration Test», Packag. Technol. Sci., vol. 30, fasc. 7, pp. 331–345, lug. 2017, doi: 10.1002/pts.2303.
    """
    def zheng_transform_leptokurtic(x, a, b):
        """zheng polynomial transformation for leptokurtic processes."""
        return np.exp(a*x)-np.exp(-b*x)
    
    def zheng_transform_platykurtic(x, a, b):
        """zheng polynomial transformation for platykurtic processes."""
        return np.log(a*x+np.sqrt((b*x)**2+1))
    
    def zheng_obj_fun(var, tranform_fun):
        """
        Objective function to minimize the absolute error in kurtosis and skewness after transformation.
        """
        nonlocal x, pdf_x, input_rms, input_kurtosis, input_skewness
        if len(var)==1:
            z = tranform_fun(grid, var, var)
        else:    
            z = tranform_fun(grid, var[0], var[1])
        
        ## * FOR EQUIVALENCE TO TIMESERIES TEST
        # zt = cubic_transform(x, var)
        # time_kurt = sp.stats.kurtosis(zt) + 3

        transformed_kurtosis = np.trapz(pdf_x*z**4,grid)/np.trapz(pdf_x*z**2,grid)**2
        transformed_skewness = np.trapz(pdf_x*z**3,grid)/np.trapz(pdf_x*z**2,grid)**(3/2)
        obj_val = np.abs(transformed_kurtosis - input_kurtosis) + np.abs(transformed_skewness - input_skewness)
        return obj_val
    
    input_rms = np.std(signal) 
    x = signal / np.max(np.abs(signal)) # input signal needs to be normalized to a N[0,1] process
    _, grid, pdf_x = _get_signal_statistics(x)
    
    if input_kurtosis>3:
        transform_function = zheng_transform_leptokurtic
        init_approx = 3.48*(input_kurtosis-3)**0.171-2 # only for a==b -> skewness = 0
    else:
        transform_function = zheng_transform_platykurtic
        init_approx = 2.36*np.exp(3-input_kurtosis)-2 **0.171-2 # only for a==b -> skewness = 0
    
    if input_skewness:
        optim_res = sp.optimize.minimize(zheng_obj_fun, args = transform_function, x0=[2, 2], bounds=[(0, 10), (1, 3)])
        a = optim_res.x[0]
        b = optim_res.x[1]
        
    else:
    # * Apoximated formulation a = b only used as initial guess for optimizator
        optim_res = sp.optimize.minimize(zheng_obj_fun, args = transform_function, x0=init_approx, bounds=[(0, 10)])
        a = optim_res.x[0]
        b = a
        
    ng_signal = transform_function(x, a, b)    
    output_rms = np.std(ng_signal)
    ng_signal = ng_signal*(input_rms/output_rms) # bring back rms to original value
    opt_params = {'a': optim_res.x[0], 'b': optim_res.x[1]}
    return ng_signal, opt_params
    
def get_sarkani(signal:np.ndarray, input_kurtosis:float, input_skewness:float = None): 
    """
    Applies Sarkani's transformation to achieve a desired kurtosis and skewness.

    Parameters
    ----------
        signal (np.ndarray): 
            Input Gaussian signal.
        input_kurtosis (float): 
            Target kurtosis.
        input_skewness (float, optional): 
            Target skewness. If None, assumes symmetry.

    Returns
    -------
        Tuple[np.ndarray, dict]: 
            Transformed signal and optimal parameters.
    Source
    ------
    S. Sarkani, D. P. Kihl, e J. E. Beach, «Fatigue of welded joints under narrowband non-Gaussian loadings», Probabilistic Eng. Mech., vol. 9, fasc. 3, pp. 179–190, gen. 1994, doi: 10.1016/0266-8920(94)90003-5.
    """
    def sarkani_transform(x, beta, n):
        """Sarkani's nonlinear transformation."""
        return np.exp(beta*x)-np.exp(-n*x)
    
    ## * OBJECTIVE FUNCTION WITH TIMEHISTORY
    # def sarkani_time_obj_fun(var):
    #     nonlocal x, sx, input_kurtosis
    #     beta = var[0]
    #     n = var[1]
    #     # C = np.sqrt(1+(2**(1/2/(n+1)*n*sp.special.gamma(n/2)*sx**(n-1)))*beta/(np.sqrt(np.pi))+(2**n*sp.special.gamma(n+0.5)*sx**(2*(n-1)))*beta**2/((np.sqrt(np.pi))))
    #     z = sarkani_transform(x, beta, n)
    #     obj_val = np.abs(sp.stats.kurtosis(z) + 3 - input_kurtosis)
    #     return obj_val
    
    #  * OBJECTIVE FUNCTION WITH PDF
    def sarkani_obj_fun(var):
        """
        Objective function to minimize the absolute error in kurtosis and skewness after transformation.
        """
        nonlocal x,input_rms, grid, pdf_x, input_kurtosis
        beta = var[0]
        n = var[1]
        z = sarkani_transform(grid, beta, n)
        
        ## * FOR EQUIVALENCE TO TIMESERIES TEST
        # zt = sarkani_transform(x, beta, n)
        # time_kurt = sp.stats.kurtosis(zt) + 3

        transformed_kurtosis = np.trapz(pdf_x*z**4,grid)/np.trapz(pdf_x*z**2,grid)**2
        obj_val = np.abs(transformed_kurtosis - input_kurtosis)
        return obj_val
    
    if input_skewness:
        raise UserWarning('Skewness correction for sarkani method is not available. Please, set skewness = 0 or consider using another method')

    x = signal
    input_rms, grid, pdf_x = _get_signal_statistics(x)
    optim_res = sp.optimize.minimize(sarkani_obj_fun, x0=[2, 2], bounds=[(0, 5), (0, 5)])
    beta = optim_res.x[0]
    n = optim_res.x[1]
    ng_signal = sarkani_transform(signal, beta, n)     
    ng_signal = ng_signal*(input_rms/np.std(ng_signal))
    opt_params = {'beta': optim_res.x[0], 'n': optim_res.x[1]}
    return ng_signal, opt_params

def get_zmnl(signal:np.ndarray, fs:float, input_kurtosis:float, input_skewness:float = None): 
    """
    Applies the improved non linear transformation to achieve a desired kurtosis.

    Parameters
    ----------
        signal (np.ndarray): 
            Input Gaussian signal.
        fs (float):
            Sampling frequency of the signal     
        input_kurtosis (float): 
            Target kurtosis.
        input_skewness (float, optional): 
            Target skewness. If None, assumes symmetry.

    Returns
    -------
        Tuple[np.ndarray, dict]: 
            Transformed signal and optimal parameters.
    Source
    ------
    G. Wise, A. Traganitis, e J. Thomas, «The effect of a memoryless nonlinearity on the spectrum of a random process», IEEE Trans. Inf. Theory, vol. 23, fasc. 1, pp. 84–89, gen. 1977, doi: 10.1109/TIT.1977.1055658.
    """
    def zengh_transform(x, beta, n, fs):
        """Improved nonlinear transformation."""
        z = (x+beta*np.sign(x)*np.abs(x)**n) 
        z = z*(input_rms/np.std(z))        
        Xts_woShift = (1/fs)*np.fft.fft(z)
        Nos         = np.ceil((len(z)+1)/2)           
        Xos = Xts_woShift[0:int(Nos)] 
        transform_FFT_phi = np.angle(Xos)
        z_amplitude_coherent= np.fft.irfft((FFT_amplitudes*np.exp(transform_FFT_phi*1j))*fs)
        return z_amplitude_coherent
    
    # * OBJECTIVE FUNCTION WITH TIMEHISTORY
    def zmnl_obj_fun(var):
        """
        Objective function to minimize the absolute error in kurtosis and skewness after transformation.
        """
        nonlocal signal, FFT_amplitudes, fs, input_kurtosis, input_rms
        beta = var[0]
        n = var[1]
        z = zengh_transform(signal, beta, n, fs)
        tranform_kurtosis = sp.stats.kurtosis(z) + 3
        obj_val = np.abs(tranform_kurtosis - input_kurtosis)
        return obj_val
    
    if input_skewness:
        raise UserWarning('Skewness correction for the improved non-linear method is not available. Please, set skewness = 0 or consider using another method')
    
    input_rms = np.std(signal)
    Xts_woShift = (1/fs)*np.fft.fft(signal)
    Nos         = np.ceil((len(signal)+1)/2)           
    Xos = Xts_woShift[0:int(Nos)] 
    FFT_amplitudes = np.abs(Xos)
    optim_res = sp.optimize.minimize(zmnl_obj_fun, x0=[2, 2], bounds=[(0, 5), (0, 5)])
    beta = optim_res.x[0]
    n = optim_res.x[1]
    ng_signal = zengh_transform(signal, beta, n)     
    ng_signal = ng_signal*(input_rms/np.std(ng_signal))
    opt_params = {'beta': optim_res.x[0], 'n': optim_res.x[1]}
    return ng_signal, opt_params

def get_steinwolf(
    signal:np.ndarray, 
    fs:float, 
    input_kurtosis:float, 
    input_skewness:float = None
    ): #! FIX: INCONSISTENT AND WRONG RESULTS  
    """
    Applies the Steinwolf phase transformation to achieve a desired kurtosis.

    Parameters
    ----------
        signal (np.ndarray): 
            Input Gaussian signal.
        fs (float):
            Sampling frequency of the signal     
        input_kurtosis (float): 
            Target kurtosis.
        input_skewness (float, optional): 
            Target skewness. If None, assumes symmetry.

    Returns
    -------
        Tuple[np.ndarray, dict]: 
            Transformed signal and optimal parameters.
    Source
    ------
    A. Steinwolf, «Random vibration testing with kurtosis control by IFFT phase manipulation», Mech. Syst. Signal Process., vol. 28, pp. 561–573, apr. 2012, doi: 10.1016/j.ymssp.2011.11.001.
    """    
    def get_Sz(A, phi): #! NOT TO BE USED (WIP)
        """Returns skewness obtained from the Fourier transform of the signal."""
        N = len(A)
        Sz = 0.5 * np.sum(A**2)
        term1 = (3/4) * np.sum([A[j] * A[k]**2 * np.cos(phi[j] - 2*phi[k]) for k in range(N) for j in range(2*k, N)])
        term2 = (3/2) * np.sum([A[j] * A[k] * A[m] * np.cos(phi[m] - phi[j] - phi[k])
                        for j in range(N) for k in range(j+1, N) for m in range(j+k, N)])
        return Sz * (term1 + term2)

    def get_Kz(A, phi): #! NOT TO BE USED (WIP)
        """Returns kurtosis obtained from the Fourier transform of the signal."""
        N = len(A)
        Kz = np.sum(A**2)**(-2)
        term1 = (3/2) * np.sum(A**4) + 2 * np.sum([A[j]**3 * np.cos(phi[j] - 3*phi[k]) 
                            for k in range(N) for j in range(3*k, N)])
        term2 = 6 * np.sum([A[j] * A[k] * A[n]**2 * np.cos(phi[j] - phi[k] - 2*phi[n]) 
                        for k in range(N) for n in range(N) if n != k for j in range(k+2*n, N)])
        term3 = 6 * np.sum([A[j] * A[k] * A[n]**2 * np.cos(phi[j] + phi[k] - 2*phi[n])
                        for n in range(N) for j in range(N) for k in range(2*n-j,N) if j<k])
        term4 = 12 * np.sum([A[j] * A[k] * A[n] * A[m] * np.cos(phi[j] + phi[k] - phi[n] - phi[m])
                        for j in range(N) for k in range(j+1, N) for n in range(j+1, N) for m in range(n+1, N) if j+k == n+m])
        term5 = 12 * np.sum([A[j] * A[k] * A[n] * A[m] * np.cos(phi[j] + phi[k] + phi[n] - phi[m])
                        for j in range(N) for k in range(j+1, N) for n in range(k+1, N) for m in range(j+k+n, N)])
                
        return 3+ Kz * (term1 + term2 + term3 + term4 + term5)
        
    def steinwolf_transform(a, b, deterministic_armonics_id, fs):
        """Steinwolf phase transformation non-gaussian signal builder."""
        deterministic_armonics_id = deterministic_armonics_id.astype(int)
        E = np.zeros(len(deterministic_armonics_id))
        D = np.zeros(len(deterministic_armonics_id))
        for idx, k in enumerate(deterministic_armonics_id):
            E[idx] = np.sum([a[n]*(a[j]**2-b[j]**2)-2*b[n]*a[j]*b[j]
                             for n in range(N) if n != k for j in range(k+2*n, N)])
            D[idx] = np.sum([b[n]*(a[j]**2-b[j]**2)+2*a[j]*a[n]*b[j] 
                             for n in range(N) if n != k for j in range(k+2*n, N)])
        bk = A[deterministic_armonics_id]*np.sqrt(D**2/(D**2+E**2))*np.sign(D)
        ak = np.sqrt(A[deterministic_armonics_id]**2-bk**2)*np.sign(E)
        a[deterministic_armonics_id] = ak
        b[deterministic_armonics_id] = bk
        dt = 1/fs
        Xos = np.array([complex(ai,bi) for ai, bi in zip(a, b)])
        x = np.fft.irfft(Xos/dt)
        return x
        
    def steinwolf_obj_fun(params,fs):
        """
        Objective function to minimize the absolute error in kurtosis and skewness after transformation.
        """
        nonlocal a, b, N, input_kurtosis
        n_deterministic_frequencies = params[0]
        deterministic_armonics_id = np.random.randint(np.floor(params[1]),np.ceil(params[2]),n_deterministic_frequencies.astype(int)).astype(int)
        x = steinwolf_transform(a, b, deterministic_armonics_id, fs)
        transformed_kurtosis = sp.stats.kurtosis(x) + 3
        return np.abs(transformed_kurtosis - input_kurtosis)
    
    if input_skewness:
        raise UserWarning('Skewness correction for the smallwood method is not available. Please, set skewness = 0 or consider using another method')

    input_rms = np.std(signal)    
    Xts_woShift = (1/fs)*np.fft.fft(signal)
    Nos         = np.ceil((len(signal)+1)/2)           
    Xos = Xts_woShift[0:int(Nos)] 
    A = np.abs(Xos)
    phi = np.angle(Xos)
    N = len(A)
    a = A*np.cos(phi)
    b = -A*np.sin(phi)
    
    # f_Xos = np.linspace(0,fs/2,N)
    constraint = {'type': 'ineq', 'fun': lambda x : x[2]-x[1]}
    optim_res = sp.optimize.minimize(steinwolf_obj_fun, args = (fs), x0=[1, 0,1], bounds=[(1,N//2),(1,N-2),(2,N-1)], constraints=[constraint])
    
    ng_signal = steinwolf_transform(a, b, optim_res.x)
    ng_signal*(input_rms/np.std(ng_signal))
    opt_params = {'a': optim_res.x[1],'b': optim_res.x[1], 'deterministic_armonics_id': optim_res.x[2]}
    return ng_signal, opt_params

def get_smallwood(
        fpsd:np.ndarray,
        psd: np.ndarray, 
        T: float, 
        input_kurtosis: float, 
        input_skewness: float = None, 
        seed: int = None
        ):# ! FIX: NOT WORKING 
    """
    Uses the Smallwood filtered poisson shot to achieve a desired kurtosis.

    Parameters
    ----------
        fpsd (np.ndarray): 
            Frequency vector of the power spectral density
        psd (np.ndarray):
            Power density vector of the power spectral density
        T (float)
        input_kurtosis (float): 
            Target kurtosis.
        input_skewness (float, optional): 
            Target skewness. If None, assumes symmetry.
        seed (int):
            Seed for the random generator for the Poisson shot times and amplitudes.
            
    Returns
    -------
        Tuple[np.ndarray, dict]: 
            Transformed signal and optimal parameters.
    Source
    ------
    D. O. Smallwood, «Generation of Stationary Non-Gaussian Time Histories with a Specified Cross-spectral Density», Shock Vib., vol. 4, fasc. 5–6, pp. 361–377, 1997, doi: 10.1155/1997/713593.
    """ 
    def get_shot_noise_smallwood(A, lam, p, I, N, seed = None):
        nonlocal psd, fs
        N = round(N)
        h_t = get_psd_impulse_response(psd, fs, N)
        if seed: np.random.seed(seed=seed)
        tau = np.round(np.random.exponential(scale=lam, size=round(N/I))*fs).astype(int)
        Ax = A * np.sign(np.random.binomial(size=round(N/I), n=1, p= p)-0.5)
        np.random.seed(None)
        shot_impulse = np.zeros(len(h_t))
        for tau_k, Ai in zip(np.cumsum(tau),Ax):
            if tau_k <len(h_t):
                shot_impulse[tau_k] = Ai
        shot_noise = np.convolve(shot_impulse,h_t,'same')
        return shot_noise
    
    def smallwood_opt_fun(params, seed):
        """
        Objective function to minimize the absolute error in kurtosis and skewness after transformation.
        """
        nonlocal p, input_kurtosis, N
        A = 1
        lam = params[0]
        I = params[1]
        ng_signal = get_shot_noise_smallwood(A, lam, p, I, N, seed=seed)
        transformed_kurtosis = sp.stats.kurtosis(ng_signal) + 3
        return np.abs(transformed_kurtosis - input_kurtosis)
    
    if input_skewness:
        raise UserWarning('Skewness correction for the smallwood method is not available. Please, set skewness = 0 or consider using another method')
    
    if seed is None: seed = np.random.randint(1e6,1e7-1) 
    fs = 2*fpsd[-1]
    N = round(T*fs)
    inp_rms = np.sqrt(np.trapz(psd,fpsd))
    h_t = get_psd_impulse_response(fpsd, psd, N)
    h2 = np.sum((h_t)**2)/fs
    h4 = np.sum((h_t)**4)/fs
    lam_0 = h4/h2**2*(1/(input_kurtosis**4-3))
    p = 0.5
    optim_res = sp.optimize.minimize(smallwood_opt_fun,args = (seed), x0=[lam_0, 1.5], bounds=[(0.001,100), (1,2)])
    ng_signal = get_shot_noise_smallwood(A = 1, lam = optim_res.x[0], N=N, p = p, I = optim_res.x[1])
    ng_signal = ng_signal*(inp_rms/np.std(ng_signal))
    opt_params = {'lam': optim_res.x[0], 'I': optim_res.x[1]}
    return ng_signal, opt_params

def get_vanbaren(
            fpsd:np.ndarray,
            psd: np.ndarray, 
            T: float, 
            input_kurtosis: float, 
            input_skewness: float = None, 
            seed: int = None
            ):
    """
    Uses the Vanbaren filtered poisson shot to achieve a desired kurtosis.

    Parameters
    ----------
        fpsd (np.ndarray): 
            Frequency vector of the power spectral density
        psd (np.ndarray):
            Power density vector of the power spectral density
        T (float)
        input_kurtosis (float): 
            Target kurtosis.
        input_skewness (float, optional): 
            Target skewness. If None, assumes symmetry.
        seed (int):
            Seed for the random generator for the Poisson shot times and amplitudes.
    Returns
    -------
        Tuple[np.ndarray, dict]: 
            Transformed signal and optimal parameters.
    Source
    ------
    P. Van Baren, «System and method for simultaneously controlling spectrum and kurtosis of a random vibration», US20070185620
    """ 
    def get_shot_noise_vanbaren(alpha, h_t, seed = None):
        N = len(h_t)
        gamma = 1/alpha
        if seed: np.random.seed(seed=seed)
        beta = np.random.exponential(scale=1/(alpha-1), size=N)
        shot_impulse = np.abs(np.random.randn(N))<gamma
        np.random.seed(seed=None)
        shot_noise = np.convolve(h_t,beta*shot_impulse, 'same')
        return shot_noise
    
    def vanbaren_opt_fun(param, h_t, seed):
        """
        Objective function to minimize the absolute error in kurtosis and skewness after transformation.
        """
        alpha = param[0]
        ng_signal =get_shot_noise_vanbaren(alpha, h_t,  seed)
        transformed_kurtosis = sp.stats.kurtosis(ng_signal) + 3
        transformed_skewness = sp.stats.skew(ng_signal)
        
        return np.abs(transformed_kurtosis - input_kurtosis) + np.abs(transformed_skewness)

    if input_skewness:
        raise UserWarning('Skewness correction for the smallwood method is not available. Please, set skewness = 0 or consider using another method')

    if seed is None: seed = np.random.randint(1e6,1e7-1) 
    fs = 2*fpsd[-1]
    N = round(T*fs)
    inp_rms = np.sqrt(np.trapz(psd,fpsd))
    # _, gauss_signal = get_stationary_gaussian(fpsd, psd/inp_rms**2, T)
    h_t = get_psd_impulse_response(fpsd, psd, N//2)
    
    optim_res = sp.optimize.minimize(vanbaren_opt_fun, method = 'Nelder-Mead', args = (h_t, seed), x0=[10], bounds=[(1.001,100)], options = {'disp': True})
    shot_noise = get_shot_noise_vanbaren(alpha = optim_res.x[0], h_t = h_t, seed = seed)
    ng_signal = shot_noise
    ng_signal = ng_signal*(inp_rms/np.std(ng_signal))
    opt_params = {'alpha': optim_res.x[0]}
    return ng_signal, opt_params


class StationaryNonGaussian(SingleChanSignal):
    """
    Generates a stationary non-Gaussian signal with specified skewness and kurtosis
    using a transformation method applied to a Gaussian signal with a target PSD.
    StationaryNonGaussian is a child class of SingleChanSignal.
    
    Parameters
    ----------
    fpsd: np.ndarray
        frequency vector of the power spectral density
    psd: np.ndarray
        power density vector of the power spectral density
    T: float
        time length of the signal
    kurtosis: int
        input kurtosis for the transformation model
    dfpsd: float
        frequency discretization of the psd to be stored 
    skewness: int
        input kurtosis for the transformation model
    method: str
        tranformation method to be used for generating the non-gaussian signal
    params: dict
        optional parameters for the transformation model if optimization wants to be skipped 
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
        self,
        fpsd: np.ndarray,
        psd: np.ndarray,
        T: float,
        kurtosis: int,
        dfpsd: float = 0.5,
        skewness: int = 0,
        method: str = 'winter',
        params=None,
        name: str = "",
        var: str = 'x',
        unit: str = "$m/s^2$",
        seed: int = None,
        interp: str = 'lin'
    ):
        method = method.lower()
        interp = interp.lower()

        # Define interpolation methods
        interps = {
            'lin': lin_interp_psd,
            'log': log_interp_psd
        }

        # Validate interpolation method
        method_existance_check(interp, interps)

        if len(fpsd) != len(psd):
            raise AttributeError('The frequency and PSD vectors must have the same length.')

        self.fpsd = fpsd
        self.psd = psd
        self.T = T
        self.fs = 2 * self.fpsd[-1]

        # Generate the non-Gaussian signal
        self.transform_params, self.x = self._get_nongaussian_timehistory(
            seed=seed,
            method=method,
            skewness=skewness,
            kurtosis=kurtosis,
            params=params
        )

        self.signal_type = f'Stationary - NonGaussian ({method})'

        super().__init__(
            x=self.x,
            fs=self.fs,
            dfpsd=dfpsd,
            name=name,
            var=var,
            unit=unit,
            signal_type=self.signal_type
        )

    def _get_gaussian_timehistory(self, seed=None) -> np.ndarray:
        """
        Internal function called by __init__
        Generate a stationary Gaussian signal from the given PSD.

        Returns:
            np.ndarray: The generated Gaussian signal.
        """
        _, signal = get_stationary_gaussian(self.fpsd, self.psd, self.T, seed)
        return signal

    def _get_nongaussian_timehistory(
        self,
        skewness: float,
        kurtosis: float,
        method: str = 'winter',
        params=None,
        seed=None
    ) -> tuple:
        """
        Internal function called by __init__
        Transforms a Gaussian signal into a non-Gaussian signal using the specified method.

        Returns:
            tuple: (parameters used for the transformation, transformed signal)
        """
        # Generate Gaussian signal
        gaussian_signal = self._get_gaussian_timehistory(seed=seed)

        # Define transformation methods
        methods = {
            'winter': get_winterstein,
            'cubic': get_cubic_polinomial,
            'zheng': get_zheng,
            'sarkani': get_sarkani,
            'zmnl': get_zmnl,
            'steinwolf': get_steinwolf,
            'smallwood': get_smallwood,
            'vanbaren': get_vanbaren
        }

        # Validate method
        method_existance_check(method, methods)

        print(
            f'Estimating non-Gaussian signal from parameters '
            f'(skew = {skewness}, kurtosis = {kurtosis}) using "{method}" method.'
        )

        if method in ['zmnl', 'steinwolf']:
            signal, opt_params = methods[method](
                gaussian_signal,
                input_skewness=skewness,
                input_kurtosis=kurtosis,
                fs=self.fs,
                params=params
            )
        elif method in ['vanbaren', 'smallwood']:
            signal, opt_params = methods[method](
                self.fpsd,
                self.psd,
                self.T,
                input_skewness=skewness,
                input_kurtosis=kurtosis,
                params=params
            )
        else:
            signal, opt_params = methods[method](
                gaussian_signal,
                input_skewness=skewness,
                input_kurtosis=kurtosis,
                params=params
            )

        # Output statistics
        output_skew = sp.stats.skew(signal)
        output_kurt = sp.stats.kurtosis(signal) + 3
        print(
            f'Non-Gaussian signal generated with "{method}" method: '
            f'skew = {output_skew:.1f}, kurt = {output_kurt:.1f}'
        )

        return opt_params, signal       


if __name__ == "__main__":
    pass
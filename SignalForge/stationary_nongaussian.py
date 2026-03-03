from .utils import *
from .single_chan_signal import SingleChanSignal

'''
supporting functions / 'static methods' (no need of self object)
- get_winterstein:             calculate non-gaussian signal using the Winterstein method
- get_cubic_polynomial:        calculate non-gaussian signal using the cubic polymomial method
- get_zheng:                   calculate non-gaussian signal using the hyperbolic method
- get_sarkani:                 calculate non-gaussian signal using the Sarkani method
- get_zmnl:                    calculate non-gaussian signal using the improved non-linear method
- get_steinwolf:               calculate non-gaussian signal using the Steinwolf method
- get_smallwood:               calculate non-gaussian signal using the Smallwood shotnoise method
- get_vanbaren:                calculate non-gaussian signal using the Vanbaren method

'''


def _get_signal_statistics(signal: np.ndarray) -> tuple:
    """
    Internal function to compute basic statistics and theoretical Gaussian PDF for a signal.

    Parameters
    ----------
        signal : np.ndarray
            Input signal array.

    Returns
    ----------
        rms : float
            Root mean square (standard deviation) of the signal.
        grid : np.ndarray 
            Grid over which the PDF is evaluated.
        pdf_x : np.ndarray 
            Bestfit Gaussian PDF values evaluated over the grid.
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
    params: float = None
) -> tuple:
    """
    Applies the Winterstein method to generate a non-Gaussian signal
    with specified skewness and kurtosis from a Gaussian input.

    Parameters
    ----------
        signal : np.ndarray
            Input signal.
        input_skewness : float 
            Desired skewness of the output signal.
        input_kurtosis : float     
            Desired kurtosis of the output signal.
        params : float
            Optional dictionary of parameters. Currently supports:
            - 2 or 3, for second or third order Winterstein transformation.

    Returns
    ---------
        ng_signal : np.ndarray 
            Transformed signal
        opt_params : dict
            Dictionary of optimal parameters
            
    Source
    ------
    S. R. Winerstein, «Nonlinear Vibration Models for Extremes and Fatigue», J. Eng. Mech., vol. 114, fasc. 10, pp. 1772-1790, ott. 1988, doi: 10.1061/(ASCE)0733-9399(1988)114:10(1772)
    """
    
    if input_kurtosis > 12:
        print(
            'Kurtosis requested may be too high for correct representation via '
            'Winterstein method. Consider using alternative methods.'
        )

    K = input_kurtosis + np.spacing(input_kurtosis)
    S = input_skewness

    # Normalize input signal to zero-mean, unit-variance
    input_rms = np.std(signal)
    x = signal / input_rms

    # Choose transformation order
    if params is not None and params == 2:
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
    output_rms = np.std(ng_signal)
    ng_signal *= input_rms / output_rms

    opt_params = None
    return ng_signal, opt_params

def get_cubic_polynomial(signal: np.ndarray, input_kurtosis: float, input_skewness: float = 0, params=None):
    """
    Applies a cubic polynomial transformation to a Gaussian signal to match 
    specified skewness and kurtosis.

    Parameters
    ----------
        signal : np.ndarray
            Input Gaussian signal.
        input_kurtosis : float
            Target excess kurtosis (e.g. 0 for Gaussian, >0 for leptokurtic).
        input_skewness : float
            Target skewness.
        params : dict {'a': [a0, a1, a2, a3]}, optional
            Bypass optimization with given parameters.

    Returns
    -------
        ng_signal : np.ndarray
            Transformed non-Gaussian signal, rescaled to match input RMS.
        opt_params : dict
            Dictionary of optimal parameters {'a': [a0, a1, a2, a3]}.
    """

    def cubic_transform(x, a):
        """y = a0 + a1*x + a2*x^2 + a3*x^3"""
        return a[0] + a[1]*x + a[2]*x**2 + a[3]*x**3

    def cubic_transform_deriv(x, a):
        """dy/dx = a1 + 2*a2*x + 3*a3*x^2"""
        return a[1] + 2*a[2]*x + 3*a[3]*x**2

    def nonlinear_constraint(a):
        """
        Monotonicity condition: |a2| < sqrt(3*a1*a3)
        Equivalent to: 3*a1*a3 - a2^2 >= 0
        """
        a1, a2, a3 = a[1], a[2], a[3]
        return 3*a1*a3 - a2**2

    def cubic_obj_fun(var):
        z = cubic_transform(grid, var)

        dydx = np.abs(cubic_transform_deriv(grid, var))
        dydx = np.where(dydx < 1e-10, 1e-10, dydx)

        pdf_y = pdf_x / dydx

        # Sort z for trapz integration (transform may reorder grid)
        sort_idx = np.argsort(z)
        z_s = z[sort_idx]
        pdf_y_s = pdf_y[sort_idx]

        norm = np.trapz(pdf_y_s, z_s)
        if norm <= 0:
            return 1e6
        pdf_y_s /= norm

        mean_y   = np.trapz(pdf_y_s * z_s, z_s)
        var_y    = np.trapz(pdf_y_s * (z_s - mean_y)**2, z_s)
        skew_y   = np.trapz(pdf_y_s * (z_s - mean_y)**3, z_s) / var_y**(3/2)
        kurt_y   = np.trapz(pdf_y_s * (z_s - mean_y)**4, z_s) / var_y**2 - 3  # excess kurtosis

        if np.allclose(input_kurtosis,3):
            obj_val = np.abs(skew_y - input_skewness)
        else:
            obj_val = np.abs(kurt_y - input_kurtosis) + np.abs(skew_y - input_skewness)
        return obj_val

    # Standardize input signal
    input_rms = np.std(signal)
    x = signal / input_rms  # unit variance Gaussian

    # Grid and Gaussian pdf on standardized signal
    grid = np.linspace(x.min(), x.max(), 1000)
    pdf_x = sp.stats.norm.pdf(grid, loc=0, scale=1)

    if params is not None:
        try:
            ng_signal = cubic_transform(x, params['a'])
            ng_signal *= input_rms / np.std(ng_signal)
            return ng_signal, params
        except (KeyError, TypeError):
            raise KeyError("'params' must be {'a': [a0, a1, a2, a3]}")

    constraint = {'type': 'ineq', 'fun': nonlinear_constraint}
    bounds = [(-10, 10),   # a0: offset
            (1e-3, 10),  # a1: must be > 0
            (-10, 10),   # a2: free but constrained
            (1e-3, 10)]  # a3: must be > 0

    optim_res = sp.optimize.minimize(
        cubic_obj_fun,
        x0=[0.0, 1.0, 0.0, 0.1],   # Start near identity transform
        bounds=bounds,
        constraints=[constraint],
        method='SLSQP',
        options={'ftol': 1e-9, 'maxiter': 1000}
    )

    if not optim_res.success:
        print(f"Warning: optimization did not converge. Message: {optim_res.message}")

    ng_signal = cubic_transform(x, optim_res.x)
    ng_signal *= input_rms / np.std(ng_signal)  # Rescale to original RMS

    opt_params = {'a': optim_res.x}
    return ng_signal, opt_params

def get_zheng(signal: np.ndarray, input_kurtosis: float, input_skewness: float = 0, params=None):
    """
    Applies Zheng's transformation to achieve a desired kurtosis and skewness.

    Parameters
    ----------
        signal : np.ndarray
            Input Gaussian signal.
        input_kurtosis : float
            Target excess kurtosis (e.g. 0 for Gaussian).
        input_skewness : float, optional
            Target skewness. Default is 0 (symmetric).
        params : dict {'a': float, 'b': float}, optional
            Transformation parameters to bypass the optimization procedure.

    Returns
    -------
        ng_signal : np.ndarray
            Transformed signal rescaled to original RMS.
        opt_params : dict
            Dictionary of optimal parameters {'a': float, 'b': float}.

    Source
    ------
    R. Zheng et al., Packag. Technol. Sci., vol. 30, no. 7, pp. 331-345, 2017.
    """

    def zheng_transform_leptokurtic(x, a, b):
        """Zheng transformation for leptokurtic processes (kurtosis > 3)."""
        return np.exp(a * x) - np.exp(-b * x)

    def zheng_transform_leptokurtic_deriv(x, a, b):
        return a * np.exp(a * x) + b * np.exp(-b * x)

    def zheng_transform_platykurtic(x, a, b):
        """Zheng transformation for platykurtic processes (kurtosis < 3)."""
        return np.log(a * x + np.sqrt((b * x)**2 + 1))

    def zheng_transform_platykurtic_deriv(x, a, b):
        inner = a * x + np.sqrt((b * x)**2 + 1)
        d_inner = a + (b**2 * x) / np.sqrt((b * x)**2 + 1)
        return d_inner / inner

    def zheng_obj_fun(var, transform_fun, transform_deriv):
        a, b = (var[0], var[0]) if len(var) == 1 else (var[0], var[1])

        z = transform_fun(grid, a, b)

        dydx = np.abs(transform_deriv(grid, a, b))
        dydx = np.where(dydx < 1e-10, 1e-10, dydx)

        pdf_y = pdf_x / dydx

        # Sort in case transform reorders the grid
        sort_idx = np.argsort(z)
        z_s = z[sort_idx]
        pdf_y_s = pdf_y[sort_idx]

        norm = np.trapz(pdf_y_s, z_s)
        if norm <= 0:
            return 1e6
        pdf_y_s /= norm

        mean_y = np.trapz(pdf_y_s * z_s, z_s)
        var_y  = np.trapz(pdf_y_s * (z_s - mean_y)**2, z_s)
        if var_y <= 0:
            return 1e6

        skew_y = np.trapz(pdf_y_s * (z_s - mean_y)**3, z_s) / var_y**(3/2)
        kurt_y = np.trapz(pdf_y_s * (z_s - mean_y)**4, z_s) / var_y**2 - 3  # excess kurtosis

        if np.allclose(input_kurtosis,3):
            obj_val = np.abs(skew_y - input_skewness)
        else:
            obj_val = np.abs(kurt_y - input_kurtosis) + np.abs(skew_y - input_skewness)
        return obj_val

    # Standardize to unit variance (not max normalization)
    input_rms = np.std(signal)
    x = signal / input_rms

    grid = np.linspace(x.min(), x.max(), 1000)
    pdf_x = sp.stats.norm.pdf(grid, loc=0, scale=1)

    if input_kurtosis > 3:
        transform_fun   = zheng_transform_leptokurtic
        transform_deriv = zheng_transform_leptokurtic_deriv
        init_approx = 3.48 * (input_kurtosis - 3)**0.171 - 2
    else:
        transform_fun   = zheng_transform_platykurtic
        transform_deriv = zheng_transform_platykurtic_deriv
        init_approx = 2.36 * np.exp(3 - input_kurtosis) ** 0.171 - 2  # fixed bracket bug

    if params is not None:
        try:
            ng_signal = transform_fun(x, params['a'], params['b'])
            ng_signal *= input_rms / np.std(ng_signal)
            return ng_signal, params
        except (KeyError, TypeError):
            raise KeyError("'params' must be a dict with format: {'a': float, 'b': float}")

    obj = lambda var: zheng_obj_fun(var, transform_fun, transform_deriv)

    if input_skewness != 0:
        # Two free parameters
        optim_res = sp.optimize.minimize(
            obj,
            x0=[max(init_approx, 0.1), max(init_approx, 0.1)],
            bounds=[(1e-3, 10), (1e-3, 10)],
            method='SLSQP',
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        a, b = optim_res.x
    else:
        # Symmetric case: a == b, one free parameter
        optim_res = sp.optimize.minimize(
            obj,
            x0=[max(init_approx, 0.1)],
            bounds=[(1e-3, 10)],
            method='SLSQP',
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        a = b = optim_res.x[0]

    if not optim_res.success:
        print(f"Warning: optimization did not converge. Message: {optim_res.message}")

    ng_signal = transform_fun(x, a, b)
    ng_signal *= input_rms / np.std(ng_signal)

    opt_params = {'a': a, 'b': b}
    return ng_signal, opt_params
    
def get_sarkani(signal:np.ndarray, input_kurtosis:float, input_skewness:float = 0, params = None): 
    """
    Applies Sarkani's transformation to achieve a desired kurtosis and skewness.

    Parameters
    ----------
        signal : np.ndarray
            Input Gaussian signal.
        input_kurtosis : float 
            Target kurtosis.
        input_skewness : float, optional
            Target skewness. If None, assumes symmetry.
        param : dict {'b1': float, 'b2': float}
            Transformation parameters to bypass the optimization procedure

    Returns
    -------
        ng_signal : np.ndarray 
            Transformed signal
        opt_params : dict
            Dictionary of optimal parameters
    Source
    ------
    S. Sarkani, D. P. Kihl, e J. E. Beach, «Fatigue of welded joints under narrowband non-Gaussian loadings», Probabilistic Eng. Mech., vol. 9, fasc. 3, pp. 179–190, gen. 1994, doi: 10.1016/0266-8920(94)90003-5.
    """
    def sarkani_transform(x, b1, b2):
        """Sarkani's nonlinear transformation."""
        return x + b1*np.sign(x)*np.abs(x)**b2
    
    ## * OBJECTIVE FUNCTION WITH TIMEHISTORY
    # def sarkani_time_obj_fun(var):
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
        b1 = var[0]
        b2 = var[1]
        z = sarkani_transform(grid, b1, b2)
        
        ## * FOR EQUIVALENCE TO TIMESERIES TEST
        # zt = sarkani_transform(x, beta, n)
        # time_kurt = sp.stats.kurtosis(zt) + 3

        transformed_kurtosis = np.trapz(pdf_x*z**4,grid)/np.trapz(pdf_x*z**2,grid)**2
        obj_val = np.abs(transformed_kurtosis - input_kurtosis)
        return obj_val
    
    if input_skewness:
        raise UserWarning('Skewness correction for sarkani method is not available. Please, set skewness = 0 or consider using another method')

    input_rms, grid, pdf_x = _get_signal_statistics(signal)

    if params is not None: # Optimization bypass
        try:
            ng_signal = sarkani_transform(signal, params['b1'], params['b2'])    
            ng_signal = ng_signal*(input_rms/np.std(ng_signal)) # bring back rms to original value
        except: 
            raise KeyError("The 'param' argument must be a dictionary with the format: {'b1': float, 'b2': float}")
        return ng_signal, params

    optim_res = sp.optimize.minimize(sarkani_obj_fun, x0=[2, 2], bounds=[(0, 5), (1, 5)])
    b1 = optim_res.x[0]
    b2 = optim_res.x[1]
    ng_signal = sarkani_transform(signal, b1, b2)     
    ng_signal = ng_signal*(input_rms/np.std(ng_signal))
    opt_params = {'b1': optim_res.x[0], 'b2': optim_res.x[1]}
    return ng_signal, opt_params

def get_zmnl(signal:np.ndarray, fs:float, input_kurtosis:float, input_skewness:float = 0, params = None): 
    """
    Applies the improved non linear transformation to achieve a desired kurtosis.

    Parameters
    ----------
        signal : np.ndarray 
            Input Gaussian signal.
        fs : float
            Sampling frequency of the signal     
        input_kurtosis : float 
            Target kurtosis.
        input_skewness : float, optional
            Target skewness. If None, assumes symmetry.
        param : dict {'beta': float, 'n': float}
            Transformation parameters to bypass the optimization procedure

    Returns
    -------
        ng_signal : np.ndarray 
            Transformed signal
        opt_params : dict
            Dictionary of optimal parameters
    Source
    ------
    G. Wise, A. Traganitis, e J. Thomas, «The effect of a memoryless nonlinearity on the spectrum of a random process», IEEE Trans. Inf. Theory, vol. 23, fasc. 1, pp. 84–89, gen. 1977, doi: 10.1109/TIT.1977.1055658.
    """
    def zmnl_transform(x, beta, n, fs):
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
        beta = var[0]
        n = var[1]
        z = zmnl_transform(signal, beta, n, fs)
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

    if params is not None: # Optimization bypass
        try:
            ng_signal = zmnl_transform(signal, params['beta'], params['n'])    
            ng_signal = ng_signal*(input_rms/np.std(ng_signal)) # bring back rms to original value
        except: 
            raise KeyError("The 'param' argument must be a dictionary with the format: {'beta': float, 'n': float}")
        return ng_signal, params
    
    optim_res = sp.optimize.minimize(zmnl_obj_fun, x0=[2, 2], bounds=[(0, 5), (0, 5)])
    beta = optim_res.x[0]
    n = optim_res.x[1]
    ng_signal = zmnl_transform(signal, beta, n, fs)     
    ng_signal = ng_signal*(input_rms/np.std(ng_signal))
    opt_params = {'beta': optim_res.x[0], 'n': optim_res.x[1]}
    return ng_signal, opt_params

def get_steinwolf(
    signal: np.ndarray,
    fs: float,
    input_kurtosis: float,
    input_skewness: float = None,
    params=None,
):
    """
    Applies the Steinwolf phase transformation to achieve a desired kurtosis.

    Parameters
    ----------
        signal : np.ndarray
            Input Gaussian signal.
        fs : float
            Sampling frequency of the signal
        input_kurtosis : float
            Target kurtosis (3 = Gaussian; >3 = leptokurtic / high peaks).
        input_skewness : float, optional
            Target skewness. If None, assumes symmetry.
        params : dict {'nd': int, 'seed': int}
            Transformation parameters to bypass the optimisation procedure.

    Returns
    -------
        ng_signal : np.ndarray
            Transformed signal with the desired kurtosis.
        opt_params : dict
            Dictionary of optimal parameters {'nd': int, 'seed': int}.

    Source
    ------
    A. Steinwolf, «Random vibration testing with kurtosis control by IFFT phase manipulation», Mech. Syst. Signal Process., vol. 28, pp. 561–573, apr. 2012, doi: 10.1016/j.ymssp.2011.11.001.
    """

    if not ((input_skewness is None) or (input_skewness==0)):
        raise UserWarning(
            "Skewness correction for the Steinwolf method is not available. "
            "Please set skewness=None or consider using another method."
        )

    # ------------------------------------------------------------------ #
    # Forward transform                                                    #
    # ------------------------------------------------------------------ #
    input_rms = np.std(signal)
    dt        = 1.0 / fs
    n_signal  = len(signal)

    Xts = dt * np.fft.rfft(signal)
    N   = len(Xts)

    A   = np.abs(Xts)
    phi = np.angle(Xts)

    # Cosine / sine decomposition (paper, just above Eq. 10)
    #   an = An cos φn,   bn = −An sin φn
    a0 = A * np.cos(phi)
    b0 = -A * np.sin(phi)

    # Only harmonics with meaningful energy are candidates
    relevant_harmonics = np.flatnonzero(A > (input_rms / 10))

    # ------------------------------------------------------------------ #
    # Precompute index arrays for each candidate k                        #
    # ------------------------------------------------------------------ #
    # For harmonic k, D and E sum over pairs (i, j) where:
    #   i + 2j = k,  0 <= i < N,  i != k  =>  i = k-2j,  j in [0, k//2]
    #
    # Index pairs are purely geometric (depend only on k and N) — safe to
    # cache permanently.  D and E depend on the evolving a_work/b_work and
    # must be recomputed at each step of the sequential update.

    def _precompute_indices(candidate_ks):
        """
        Precompute index arrays for each candidate harmonic k for D and E sums.

        Parameters
        ----------
        candidate_ks : array-like
            Array of harmonic indices (k) for which to precompute indices.

        Returns
        -------
        dict
            A dictionary where keys are harmonic indices (k) and values are
            tuples `(i_arr, j_arr)` containing the indices for the D and E sums.
        """
        idx_cache = {}
        for k in candidate_ks:
            k     = int(k)
            j_arr = np.arange(0, k // 2 + 1)
            i_arr = k - 2 * j_arr
            mask  = (i_arr >= 0) & (i_arr < N) & (i_arr != k)
            idx_cache[k] = (i_arr[mask], j_arr[mask])
        return idx_cache


    def _compute_DE(a, b, k, idx_cache):
        """
        Compute the D and E coefficients for a given harmonic k.

        Parameters
        ----------
        a : np.ndarray
            Cosine coefficients.
        b : np.ndarray
            Sine coefficients.
        k : int
            The harmonic index.
        idx_cache : dict
            Precomputed index cache from `_precompute_indices`.

        Returns
        -------
        tuple
            A tuple containing:
            - D : float
                The D coefficient.
            - E : float
                The E coefficient.
        """
        i_arr, j_arr = idx_cache[k]
        if len(i_arr) == 0:
            return 0.0, 0.0
        ai = a[i_arr];  bi = b[i_arr]
        aj = a[j_arr];  bj = b[j_arr]
        cross = aj ** 2 - bj ** 2
        D = float(np.sum(bi * cross + 2.0 * ai * aj * bj))
        E = float(np.sum(ai * cross - 2.0 * bi * aj * bj))
        return D, E


    # ------------------------------------------------------------------ #
    # Core transform                                                       #
    # ------------------------------------------------------------------ #
    def steinwolf_transform(a_in, b_in, det_ids, idx_cache):
        """
        Apply deterministic phase selection to harmonics in det_ids.
        D/E are recomputed from the evolving a_work, b_work at each step.
        Kurtosis-increase convention: sign(bk) = sign(D), sign(ak) = sign(E).
        """
        a_work = a_in.copy()
        b_work = b_in.copy()

        for k in det_ids:
            k    = int(k)
            D, E = _compute_DE(a_work, b_work, k, idx_cache)
            Ak   = A[k]

            denom = D ** 2 + E ** 2
            if denom == 0.0 or Ak == 0.0:
                continue

            bk = Ak * np.sqrt(D ** 2 / denom) * np.sign(D)
            ak = np.sqrt(max(Ak ** 2 - bk ** 2, 0.0)) * np.sign(E)

            a_work[k] = ak
            b_work[k] = bk

        Xos = a_work - 1j * b_work
        return np.fft.irfft(Xos / dt, n=n_signal)

    def _get_kurtosis(nd, harmonics_order, idx_cache):
        """
        Compute the kurtosis of the signal after applying the Steinwolf transform
        with a given number of deterministic harmonics.

        Parameters
        ----------
        nd : int
            The number of deterministic harmonics to use for the transform.
        harmonics_order : np.ndarray
            Array of relevant harmonic orders, typically shuffled.
        idx_cache : dict
            Precomputed index cache.

        Returns
        -------
        float
            The kurtosis of the transformed signal (Fisher's definition, i.e., excess kurtosis).
        """
        nd = max(1, min(int(nd), len(harmonics_order) - 1))
        x  = steinwolf_transform(a0, b0, harmonics_order[:nd], idx_cache)
        return sp.stats.kurtosis(x, fisher=False)

    # ------------------------------------------------------------------ #
    # Adaptive nd bounds                                                   #
    # ------------------------------------------------------------------ #
    def _find_nd_bounds(harmonics_order, idx_cache):
        """
        Determine the search interval [nd_lo, nd_hi] for the grid search.

        Strategy
        --------
        1.  Evaluate kurtosis with ALL relevant harmonics transformed → K_max.
            If the target kurtosis already exceeds K_max the signal cannot reach
            it; we return the full range and let the grid search find the best
            achievable nd.
        2.  Binary search for the smallest nd such that K(nd) >= input_kurtosis.
            This becomes nd_hi — no nd above it can be closer to the target
            (kurtosis is monotonically non-decreasing with nd).
        3.  nd_lo is always 1.

        Parameters
        ----------
        harmonics_order : np.ndarray
            Array of relevant harmonic orders, typically shuffled.
        idx_cache : dict
            Precomputed index cache for computing D and E.

        Returns
        -------
        tuple
            A tuple containing:
            - nd_lo : int
                Lower bound of the search interval for nd.
            - nd_hi : int
                Upper bound of the search interval for nd.
        """
        n_candidates = len(harmonics_order)

        # Step 1: maximum achievable kurtosis
        k_max = _get_kurtosis(n_candidates - 1, harmonics_order, idx_cache)

        if input_kurtosis >= k_max:
            # Target unreachable — search full range, best effort
            return 1, n_candidates - 1

        # Step 2: binary search for smallest nd where K(nd) >= input_kurtosis.
        # Kurtosis is monotonically non-decreasing with nd, so this is valid.
        lo, hi = 1, n_candidates - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if _get_kurtosis(mid, harmonics_order, idx_cache) >= input_kurtosis:
                hi = mid
            else:
                lo = mid + 1

        # nd_hi is the first nd that meets or exceeds the target; the optimum
        # lies in [nd_hi - 1, nd_hi] but we add a small margin for safety.
        nd_hi = min(hi + 1, n_candidates - 1)
        nd_lo = max(1, hi - 1)
        return nd_lo, nd_hi

    # ------------------------------------------------------------------ #
    # Bypass mode                                                          #
    # ------------------------------------------------------------------ #
    if params is not None:
        try:
            seed = params["seed"]
            nd   = int(params["nd"])
        except (KeyError, TypeError):
            raise KeyError(
                "The 'params' argument must be a dict with keys "
                "'nd' (int) and 'seed' (int)."
            )
        rng      = np.random.default_rng(seed=seed)
        shuffled = relevant_harmonics.copy()
        rng.shuffle(shuffled)
        idx_cache = _precompute_indices(shuffled[:nd])
        ng_signal = steinwolf_transform(a0, b0, shuffled[:nd], idx_cache)
        ng_signal = ng_signal * (input_rms / np.std(ng_signal))
        return ng_signal, params

    # ------------------------------------------------------------------ #
    # Optimisation: bounded integer grid search over nd                   #
    # ------------------------------------------------------------------ #
    seed     = int(np.random.randint(int(1e6), int(1e7) - 1))
    rng      = np.random.default_rng(seed=seed)
    shuffled = relevant_harmonics.copy()
    rng.shuffle(shuffled)

    idx_cache = _precompute_indices(shuffled)

    # Narrow the search range before the grid sweep
    nd_lo, nd_hi = _find_nd_bounds(shuffled, idx_cache)

    best_nd, best_err = nd_lo, np.inf
    for nd in range(nd_lo, nd_hi + 1):
        err = np.abs(_get_kurtosis(nd, shuffled, idx_cache) - input_kurtosis)
        if err < best_err:
            best_err = err
            best_nd  = nd

    ng_signal = steinwolf_transform(a0, b0, shuffled[:best_nd], idx_cache)
    ng_signal = ng_signal * (input_rms / np.std(ng_signal))

    opt_params = {"nd": best_nd, "seed": seed}
    return ng_signal, opt_params

def get_smallwood(
        fpsd:np.ndarray,
        psd: np.ndarray, 
        T: float, 
        input_kurtosis: float, 
        input_skewness: float = None, 
        seed: int = None,
        params = None
        ):
    """
    Uses the Smallwood filtered poisson shot to achieve a desired kurtosis.

    Parameters
    ----------
        fpsd : np.ndarray 
            Frequency vector of the power spectral density
        psd : np.ndarray
            Power density vector of the power spectral density
        T : float
            Time length of the final signal
        input_kurtosis : float 
            Target kurtosis.
        input_skewness : float, optional
            Target skewness. If None, assumes symmetry.
        seed : int
            Seed for the random generator for the Poisson shot times and amplitudes.
        param : dict {'lam': float, 'I': float, 'seed': int}
            Transformation parameters to bypass the optimization procedure
            
    Returns
    -------
        ng_signal : np.ndarray 
            Transformed signal
        opt_params : dict
            Dictionary of optimal parameters
    Source
    ------
    D. O. Smallwood, «Generation of Stationary Non-Gaussian Time Histories with a Specified Cross-spectral Density», Shock Vib., vol. 4, fasc. 5–6, pp. 361–377, 1997, doi: 10.1155/1997/713593.
    """ 
    if input_kurtosis <= 3.0:
        raise ValueError("Smallwood shot noise method requires kurtosis > 3.")

    if seed is None:
        seed = np.random.randint(int(1e6), int(1e7) - 1)

    fs  = 2.0 * fpsd[-1]
    dt  = 1.0 / fs
    N   = round(T * fs)
    K4  = input_kurtosis
    S3  = input_skewness if input_skewness is not None else 0.0

    # Target RMS from PSD
    inp_rms = np.sqrt(np.trapz(psd, fpsd))
    E_x2    = inp_rms ** 2   # target variance (signal is zero-mean)

    # --- Impulse response h(t) from PSD, Eq. (8) ---
    # get_psd_impulse_response returns h already rolled (N//2 shift)
    # and normalized so that sum(h²)*dt ≈ inp_rms²
    h_base = get_psd_impulse_response(fpsd, psd, N)

    # --- h̄_n integrals, Eq. (3a): h̄_n = ∫ h^n(t) dt ---
    h2_bar = np.sum(h_base ** 2) * dt
    h3_bar = np.sum(h_base ** 3) * dt
    h4_bar = np.sum(h_base ** 4) * dt

    # Sanity check: h2_bar should equal E[x²] = inp_rms²
    # because Parseval: ∫h²dt = ∫G_xx df = inp_rms²
    assert np.isclose(h2_bar, E_x2, rtol=0.05), (
        f"h2_bar={h2_bar:.4f} should ≈ inp_rms²={E_x2:.4f}. "
        "Check get_psd_impulse_response normalization."
    )

    # --- Closed-form parameters, Eq. (13) ---
    # lam [Hz]: mean impulse arrival rate
    lam = (h4_bar / h2_bar ** 2) * (1.0 / (K4 - 3.0))

    # Ax: impulse amplitude (Eq. 13)
    # Note: lam [1/s] * h2_bar [units²·s] = [units²], consistent with E_x2 [units²]
    Ax = np.sqrt(E_x2 / (lam * h2_bar))

    # p: probability of +Ax vs -Ax (Eq. 13), controls skewness
    if h3_bar != 0.0 and S3 != 0.0:
        p = 0.5 * (S3 * np.sqrt(lam * h2_bar ** 3) / h3_bar + 1.0)
    else:
        p = 0.5   # symmetric → zero skewness

    p = float(np.clip(p, 0.0, 1.0))

    def _generate(lam_hz, Ax, p, seed):
        """Core shot noise generator, Eqs. (6), (19), (20)."""
        rng = np.random.default_rng(seed)

        # Scale h by 1/sqrt(lam), Eq. (16)
        h_t = h_base / np.sqrt(lam_hz)

        # Poisson inter-arrival times in samples, Eq. (19)
        # lam_hz [1/s] → mean inter-arrival = fs/lam_hz [samples]
        mean_interval = fs / lam_hz
        n_impulses    = max(int(lam_hz * T * 6), 200)   # generous oversample

        arrivals = np.cumsum(
            np.round(rng.exponential(scale=mean_interval, size=n_impulses)).astype(int)
        )
        arrivals = arrivals[arrivals < N]

        # Amplitudes: +Ax with prob p, -Ax with prob (1-p), Eq. (11)
        amplitudes = np.where(rng.uniform(size=len(arrivals)) < p, Ax, -Ax)

        # Impulse train
        impulse_train = np.zeros(N)
        for k, Ai in zip(arrivals, amplitudes):
            impulse_train[k] = Ai

        # Circular convolution via FFT, consistent with rolled h_t
        shot_noise = np.fft.irfft(
            np.fft.rfft(impulse_train) * np.fft.rfft(h_t), n=N
        )
        return shot_noise

    if params is not None:
        try:
            ng = _generate(params['lam'], params.get('Ax', Ax),
                           params.get('p', p),  params.get('seed', seed))
            return ng, params
        except KeyError:
            raise KeyError("params must contain at least {'lam': float}")

    ng_signal = _generate(lam, Ax, p, seed)

    # kurtosis is scale-invariant so this rescaling does NOT affect it
    ng_signal *= inp_rms / np.std(ng_signal)

    opt_params = {'lam': lam, 'Ax': Ax, 'p': p, 'seed': seed}

    # print(f"lam            = {lam:.4f} Hz")
    # print(f"total impulses = {lam * T:.1f}  (small → non-Gaussian)")
    # print(f"Ax             = {Ax:.6f}")
    # print(f"p              = {p:.4f}")
    # print(f"achieved Kurt  = {sp.stats.kurtosis(ng_signal, fisher=False):.3f}  "
    #       f"(target={K4})")

    return ng_signal, opt_params

def get_vanbaren(
        fpsd: np.ndarray,
        psd: np.ndarray,
        T: float,
        input_kurtosis: float,
        input_skewness: float = 0,
        seed: int = None,
        params: dict = None
        ):
    """
    Uses the Van Baren filtered Poisson shot noise to achieve a desired kurtosis.

    Parameters
    ----------
        fpsd : np.ndarray
            Frequency vector of the power spectral density.
        psd : np.ndarray
            Power spectral density vector.
        T : float
            Time length of the final signal [s].
        input_kurtosis : float
            Target excess kurtosis (e.g. 3 for Gaussian).
        input_skewness : float, optional
            Target skewness. Only 0 is supported (symmetric process). Default is 0.
        seed : int, optional
            Seed for reproducibility of Poisson shot times and amplitudes.
        params : dict {'alpha': float, 'seed': int}, optional
            Bypass optimization with given parameters.

    Returns
    -------
        ng_signal : np.ndarray
            Transformed non-Gaussian signal scaled to match input RMS.
        opt_params : dict
            Dictionary of optimal parameters {'alpha': float, 'seed': int}.

    Source
    ------
    P. Van Baren, 'System and method for simultaneously controlling spectrum and kurtosis of a random vibration', US20070185620.
    """
    def get_shot_noise_vanbaren(alpha, h_t, seed=None):
        """Generate Van Baren filtered Poisson shot noise."""
        if alpha <= 1:
            raise ValueError(f"alpha must be > 1, got {alpha:.4f}")
        N = len(h_t)
        gamma = 1 / alpha
        rng = np.random.default_rng(seed)
        beta = rng.exponential(scale=1 / (alpha - 1), size=N)
        shot_impulse = np.abs(rng.standard_normal(N)) < gamma

        active_idx = np.where(shot_impulse)[0]
        active_weights = beta[active_idx]

        # Build sparse impulse signal then FFT-convolve once
        sparse_signal = np.zeros(N)
        sparse_signal[active_idx] = active_weights
        shot_noise = sp.signal.fftconvolve(h_t, sparse_signal, mode='same')
        return shot_noise

    def vanbaren_obj_fun(param, h_t, seed):
        """Objective function: minimize kurtosis error."""
        alpha = param[0]
        if alpha <= 1.0:
            return 1e6  # infeasible
        
        ng = get_shot_noise_vanbaren(alpha, h_t, seed)
        
        if np.std(ng) < 1e-10:  # degenerate signal guard
            return 1e6

        kurt = sp.stats.kurtosis(ng, fisher=False)  # excess=False -> raw kurtosis
        obj = np.abs(kurt - input_kurtosis)
        return obj

    # --- Input validation ---
    if input_skewness != 0:
        raise ValueError(
            "Skewness correction is not supported for the Van Baren method. "
            "Please set input_skewness=0 or use another method."
        )
    if input_kurtosis < 3:
        raise ValueError(
            "Van Baren shot noise targets kurtosis >= 3 (leptokurtic). "
            "For platykurtic signals, consider another method."
        )

    # --- Setup ---
    if seed is None:
        seed = np.random.randint(int(1e6), int(1e7) - 1)

    fs = 2 * fpsd[-1]
    N = round(T * fs)
    inp_rms = np.sqrt(np.trapz(psd, fpsd))
    h_t = get_psd_impulse_response(fpsd, psd, N // 2)

    # --- Bypass optimization ---
    if params is not None:
        try:
            ng_signal = get_shot_noise_vanbaren(
                alpha=params['alpha'], h_t=h_t, seed=params['seed']
            )
            ng_signal *= inp_rms / np.std(ng_signal)
            return ng_signal, params
        except (KeyError, TypeError):
            raise KeyError(
                "'params' must be a dict with format: {'alpha': float, 'seed': int}"
            )

    # --- Initial guess based on target kurtosis ---
    # Higher kurtosis -> lower alpha (more impulsive)
    alpha_init = max(1.5, 10 / (input_kurtosis - 2))

    optim_res = sp.optimize.minimize(
        vanbaren_obj_fun,
        x0=[alpha_init],
        args=(h_t, seed),
        method='Nelder-Mead',
        bounds=[(1.001, 100)],
        options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 500, 'disp': False}
    )

    if not optim_res.success:
        print(f"Warning: optimization did not converge. Message: {optim_res.message}")

    alpha_opt = optim_res.x[0]
    ng_signal = get_shot_noise_vanbaren(alpha=alpha_opt, h_t=h_t, seed=seed)
    ng_signal *= inp_rms / np.std(ng_signal)

    opt_params = {'alpha': alpha_opt, 'seed': seed}
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
        flims: list = None,
        fs : float = None,
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
        """
        Initializes a StationaryNonGaussian signal generator.

        Parameters
        ----------
        fpsd : np.ndarray
            Frequency vector of the power spectral density.
        psd : np.ndarray
            Power density vector of the power spectral density.
        T : float
            Time length of the signal.
        kurtosis : int or array-like
            Input kurtosis for the transformation model. Can be a scalar or array for banded modulation.
        flims : list, optional
            Frequency limits for banded modulation. Required if `kurtosis` is array-like.
        fs : float, optional
            Wanted sampling frequency. If None, it defaults to 2*fpsd[-1].
        dfpsd : float, optional
            Frequency discretization of the PSD to be stored. Default is 0.5.
        skewness : int, optional
            Input skewness for the transformation model. Default is 0.
        method : str, optional
            Transformation method to be used for generating the non-Gaussian signal. Default is 'winter'.
        params : dict, optional
            Optional parameters for the transformation model if optimization wants to be skipped.
        name : str, optional
            Name of the process (used for plots). Default is "".
        var : str, optional
            Name of the variable (used for plots). Default is 'x'.
        unit : str, optional
            Unit of measure of the signal (used for plots). Default is '$m/s^2$'.
        seed : int, optional
            Seed for the random generator used in non-Gaussian models. Default is None.
        interp : str, optional
            Interpolation rule for increasing resolution on given PSD ('lin' or 'log'). Default is 'lin'.
        """
        method = method.lower()
        interp = interp.lower()

        # Define interpolation methods
        interps = {
            'lin': lin_interp_psd,
            'log': log_interp_psd
        }

        # Validate interpolation method
        method_existance_check(interp, interps)
        self._interp = interp
        
        if len(fpsd) != len(psd):
            raise AttributeError('The frequency and PSD vectors must have the same length.')

        self.fpsd = np.asarray(fpsd)
        self.psd = np.asarray(psd)
        self.T = T
        if fs is None: 
            self.fs = fpsd[-1] * 2  # Nyquist frequency
        else:
            self.fs = fs
        self.N = np.round(self.T*self.fs).astype(int)
        try: 
            np.array(kurtosis)
        except:
            ValueError(f'Kurtosis value must be input in an array-like format. {type(kurtosis)} type not supported') 

        # Generate the non-Gaussian signal
        if  np.isscalar(kurtosis):
            self.transform_params, self.x = self._get_nongaussian_timehistory(
                seed=seed,
                method=method,
                skewness=skewness,
                kurtosis=kurtosis,
                params=params
            )

        elif kurtosis.ndim == 1:
            self.transform_params, self.x = self._get_banded_nongaussian_timehistory(
                seed=seed,
                method=method,
                skewness=skewness,
                kurtosis=kurtosis,
                flims = flims
            )
        else:
            ValueError(f'Something is wrong with the size of the kurtosis input') 

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

    def get_existing_methods(self):
        """
        Returns a dictionary mapping method names to their corresponding functions.

        Returns
        -------
            dict
                A dictionary where keys are method names (str) and values are
                callable functions for non-Gaussian transformation.
        """
        # Define transformation methods
        methods = {
            'winter': get_winterstein,
            'cubic': get_cubic_polynomial,
            'zheng': get_zheng,
            'sarkani': get_sarkani,
            'zmnl': get_zmnl,
            'steinwolf': get_steinwolf,
            'smallwood': get_smallwood,
            'vanbaren': get_vanbaren
            }
        return methods
    
    def _get_gaussian_timehistory(self, seed=None) -> np.ndarray:
        """
        Internal function called by __init__
        Generate a stationary Gaussian signal from the given PSD.

        Parameters
        ----------
            seed : int, optional
                Seed for random number generation.

        Returns
        -------
            np.ndarray 
                The generated stationary Gaussian signal.
        """
        _, signal = get_stationary_gaussian(self.fpsd, self.psd, self.T, fs = self.fs, seed = seed, interp= self._interp)
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

        methods = self.get_existing_methods()

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
            interp_fpsd, interp_psd = log_interp_psd(self.fpsd, self.psd, self.N//2, self.fs)
            signal, opt_params = methods[method](
                interp_fpsd,
                interp_psd,
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

    def _get_banded_nongaussian_timehistory(
        self,
        skewness: float,
        kurtosis: list,
        flims: list = None,
        method: str = 'winter',
        seed: int =None
        ):
        """
        Internal function to generate a banded non-Gaussian signal.

        This method splits the frequency spectrum into bands and applies non-Gaussian
        transformations to each band independently to achieve target kurtosis values.

        Parameters
        ----------
            skewness : float
                Target skewness for the non-Gaussian signal.
            kurtosis : list
                List of target kurtosis values, one for each frequency band.
            flims : list, optional
                Frequency limits [lower, upper] for the entire banded process.
                If None, defaults to [0, self.fs].
            method : str, optional
                The non-Gaussian transformation method to apply ('winter', 'cubic', etc.).
                Default is 'winter'.
            seed : int, optional
                Random seed for reproducibility.

        Returns
        -------
            opt_params : list
                List of dictionaries, each containing optimal parameters for the
                transformation applied to a specific band.
            out_signal : np.ndarray
                The combined non-Gaussian signal generated from all bands.
        """
        
        methods = self.get_existing_methods()
        
        # Generate Gaussian signal
        gaussian_signal = self._get_gaussian_timehistory(seed=seed)
        
        if flims is None:
            flims = [0, self.fs]
            
        print(
            f'Estimating banded non-Gaussian signal from parameters '
            f'(skew = {skewness}, kurtosis = {kurtosis}) between {flims[0]} Hz and {flims[1]} Hz using "{method}" method.'
        )
        
        n_bands = len(kurtosis)
        f_incr = (flims[1]-flims[0])/n_bands
        fl_vector = np.arange(0,n_bands)*f_incr + flims[0]
        
        out_signal = np.zeros(len(gaussian_signal))
        opt_params = []
        
        for fl, kurt in zip(fl_vector, kurtosis):
            fu = fl+f_incr
            if method in ['zmnl', 'steinwolf']:
                filt_gaussian_signal = perfect_passband_filter(gaussian_signal, fl_n = fl/self.fs, fu_n = fu/self.fs)
                signal, band_opt_params = methods[method](
                    filt_gaussian_signal,
                    input_skewness=skewness,
                    input_kurtosis=kurt,
                    fs=self.fs
                )
            elif method in ['vanbaren', 'smallwood']:
                interp_fpsd, interp_psd = log_interp_psd(self.fpsd, self.psd, self.N//2, self.fs)
                filtered_psd = np.zeros(len(interp_psd))
                mask = (interp_fpsd>fl) & (interp_fpsd<(fl+f_incr))
                filtered_psd[mask] = interp_psd[mask]
                signal, band_opt_params = methods[method](
                    interp_fpsd,
                    filtered_psd,
                    self.T,
                    input_skewness=skewness,
                    
                    input_kurtosis=kurt
                )
            else:
                filt_gaussian_signal = perfect_passband_filter(gaussian_signal, fl/self.fs, fu/self.fs)
                signal, band_opt_params = methods[method](
                    filt_gaussian_signal,
                    input_skewness=skewness,
                    input_kurtosis=kurt
                )
            print(f'Band generated (range {fl:.2f}-{fu:.2f} Hz) with kurt = {sp.stats.kurtosis(signal)+3:.2f} (expected {kurt:.2f})')
            out_signal += signal
            opt_params.append(band_opt_params) 
        
        return opt_params, out_signal

if __name__ == "__main__":
    pass
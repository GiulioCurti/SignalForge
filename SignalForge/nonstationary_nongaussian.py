import matplotlib.pyplot as plt

from .utils import *
from .single_chan_signal import SingleChanSignal

'''
supporting functions / 'static methods' (no need of self object)
- get_beta_amplitude_modulation:            adds amplitude modulated non-stationarity to gaussian signal via beta distribution
- get_rayleigh_amplitude_modulation:        adds amplitude modulated non-stationarity to gaussian signal via rayleigh distribution
- get_trapp_amplitude_modulation:           adds amplitude modulated non-stationarity to gaussian signal via gaussian distribution
- get_frequency_modulation:                 designes frequency modulate non-stationary process from given modulation function and starting psd
'''

def _add_am_modulation(carrier:np.ndarray, modulation:np.ndarray):
    """
    private function for adding amplitude modulation modulation to gaussian noise standardizing rms to the carrier 
    """
    nonstat_sign = carrier*modulation
    rms_correction = np.std(carrier) / np.std(nonstat_sign)
    nonstat_sign = nonstat_sign * rms_correction
    return nonstat_sign

def get_beta_amplitude_modulation(
    carrier: np.ndarray,
    T: float,
    input_kurtosis: float = None,
    alpha: float = None,
    beta: float = None,
    f_mod: float = None,
    seed: int = None,
    param_bounds: dict = None,
):
    """
    Apply beta-distributed amplitude modulation to a Gaussian signal.
 
    Two modes of operation
    ----------------------
        Direct
            If ``alpha``, ``beta`` and ``f_mod`` are all provided, the modulation function is
            built straight away and ``input_kurtosis`` is ignored.
        Optimize
            If any of ``alpha``, ``beta`` or ``f_mod`` is left as ``None``, the
            missing one(s) are fitted so the output kurtosis matches
            ``input_kurtosis``. Provided parameters are held fixed.
 
    Parameters
    ----------
        carrier : np.ndarray
            Input Gaussian signal.
        T : float
            Signal duration in seconds.
        input_kurtosis : float, optional
            Target kurtosis. Required whenever at least one shape parameter is
            missing; ignored in direct mode.
        alpha, beta : float, optional
            Shape parameters of the Beta(alpha, beta) amplitude distribution.
            Small alpha with large beta -> rare, spiky bursts -> high kurtosis.
        f_mod : float, optional
            Modulation frequency in Hz. Internally mapped to a
            number of control points via ``Ntw = round(2 * f_mod * T)``.
        seed : int, optional
            Seed for reproducibility. If None a random seed is drawn once and used
            consistently, so the fitted parameters and the returned modulation always
            describe the same realization.
        param_bounds : dict, optional
            Override default optimization bounds, e.g. ``{'beta': (0.1, 500)}``.
 
    Returns
    -------
        nonstat_signal : np.ndarray
            Non-stationary resulting signal.
        modulation : dict
            ``{'name', 'modulation', 'params'}`` where ``params`` reports the
            alpha/beta/f_mod actually used.
 
    Source
    ------
    D. Smallwood, "Vibration with Non-Gaussian Noise", J. IEST, vol. 52,
    no. 2, pp. 13-30, 2009, doi: 10.17764/jiet.52.2.gh0444564n8765k1.
    Smallwood controls kurtosis by adjusting both Beta parameters; here f_mod
    sets the modulation bandwidth independently of the kurtosis shape.
    """
    N = len(carrier)
    fs = N / T
    time_vector = np.linspace(0, 1, N) * T
 
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31 - 1))
 
    def get_beta_modulation(alpha, beta, f_mod, rng):
        Ntw = max(4, int(round(2 * f_mod * T)))
        t_mod = np.linspace(0, T, Ntw)
        beta_points = 2 * rng.beta(alpha, beta, Ntw)
        beta_points[[-1, 0]] = 0
        spline = sp.interpolate.CubicSpline(t_mod, beta_points)
        return spline(time_vector) / np.max(beta_points)
 
    def build(alpha, beta, f_mod):
        rng = np.random.default_rng(seed)
        modulation = get_beta_modulation(alpha, beta, f_mod, rng)
        nonstat = _add_am_modulation(carrier, modulation)
        return nonstat, modulation
 
    given = {"alpha": alpha, "beta": beta, "f_mod": f_mod}
    free = [k for k, v in given.items() if v is None]
 
    # ---------------------------------------------------------------- direct
    if not free:
        if input_kurtosis is not None:
            print(
                "alpha, beta and f_mod are all specified; input_kurtosis is ignored."
            )
        nonstat_signal, modulation = build(alpha, beta, f_mod)
        return nonstat_signal, {
            "name": "Beta amplitude modulation [-]",
            "modulation": [modulation],
            "params": {"alpha": alpha, "beta": beta, "f_mod": f_mod},
        }
 
    # -------------------------------------------------------------- optimize
    
    if input_kurtosis is None:
        raise ValueError(
            "input_kurtosis is required when alpha, beta or f_mod is not specified."
        )
    print(f'No beta distribution parameters given. Optimizing parameters for given kurtosis.')
    bounds = {
        "alpha": (0.05, 5.0),
        "beta": (0.05, 200.0),
        "f_mod": (max(2.0 / T, 0.5), fs / 40),
    }
    if param_bounds:
        bounds.update(param_bounds)
 
    def assemble(free_vals):
        p = dict(given)
        for k, v in zip(free, free_vals):
            p[k] = v
        return p["alpha"], p["beta"], p["f_mod"]
 
    def signed_err(free_vals):
        a, b, fm = assemble(np.atleast_1d(free_vals))
        rng = np.random.default_rng(seed)
        modulation = get_beta_modulation(a, b, fm, rng)
        return sp.stats.kurtosis(_add_am_modulation(carrier, modulation)) + 3 - input_kurtosis
 
    free_bounds = [bounds[k] for k in free]
 
    if len(free) == 1 and free[0] in ("alpha", "beta"):
        lo, hi = free_bounds[0]
        e_lo, e_hi = signed_err(lo), signed_err(hi)
        if e_lo * e_hi <= 0:
            best = [sp.optimize.brentq(signed_err, lo, hi, xtol=1e-2)]
        else:  # target outside reachable range -> closest endpoint
            best = [lo if abs(e_lo) < abs(e_hi) else hi]
    else:
        res = sp.optimize.differential_evolution(
            lambda v: abs(signed_err(v)), free_bounds,
            seed=seed, tol=1e-2, polish=False, maxiter=200,
        )
        best = res.x
 
    alpha, beta, f_mod = assemble(np.atleast_1d(best))
    nonstat_signal, modulation = build(alpha, beta, f_mod)
 
    achieved = sp.stats.kurtosis(nonstat_signal) + 3
    if abs(achieved - input_kurtosis) > 0.5:
        print(
            f"Target kurtosis {input_kurtosis} not reached (got {achieved:.2f}). "
            "Free another parameter, or lower alpha / raise the beta bound for higher kurtosis."
        )
    return nonstat_signal, {
        "name": "Beta amplitude modulation [-]",
        "modulation": [modulation],
        "params": {"alpha": alpha, "beta": beta, "f_mod": f_mod},
    }
    
def get_rayleigh_amplitude_modulation(
    carrier: np.ndarray,
    T: float,
    input_kurtosis: float = None,
    f_mod: float = None,
    seed: int = None,
    param_bounds: dict = None,
):
    """
    Apply Rayleigh-distributed amplitude modulation to a Gaussian signal.

    Two modes of operation
    ----------------------
        Direct
            If ``f_mod`` is provided, the modulation function is built straight away
            and ``input_kurtosis`` is ignored.
        Optimize
            If ``f_mod`` is left as ``None``, it is fitted so the output kurtosis
            matches ``input_kurtosis``.

    Parameters
    ----------
        carrier : np.ndarray
            Input Gaussian signal.
        T : float
            Signal duration in seconds.
        input_kurtosis : float, optional
            Target kurtosis. Required whenever ``f_mod`` is missing; ignored in
            direct mode.
        f_mod : float, optional
            Modulation frequency in Hz. Internally mapped to a number of control
            points via ``Ntw = round(2 * f_mod * T)``. Few control points -> slow,
            broad modulation; many control points -> rarer, spikier peaks after
            peak-normalization -> higher kurtosis.
        seed : int, optional
            Seed for reproducibility. If None a random seed is drawn once and used
            consistently, so the fitted f_mod and the returned modulation always
            describe the same realization.
        param_bounds : dict, optional
            Override the default optimization bounds, e.g. ``{'f_mod': (1.0, 200.0)}``.

    Returns
    -------
        nonstat_signal : np.ndarray
            Non-stationary resulting signal.
        modulation : dict
            ``{'name', 'modulation', 'params'}`` where ``params`` reports the f_mod
            actually used.

    Notes
    -----
    The Rayleigh scale parameter is not exposed: the modulation is normalized by its
    own peak and kurtosis is scale invariant, so the scale cannot influence the
    output kurtosis. The modulation bandwidth (``f_mod``) is the only shape knob.

    Source
    ------
    D. Smallwood, "Vibration with Non-Gaussian Noise", J. IEST, vol. 52,
    no. 2, pp. 13-30, 2009, doi: 10.17764/jiet.52.2.gh0444564n8765k1.
    """
    N = len(carrier)
    fs = N / T
    time_vector = np.linspace(0, 1, N) * T

    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31 - 1))

    def get_ray_modulation(f_mod, rng):
        Ntw = max(4, int(round(2 * f_mod * T)))
        t_mod = np.linspace(0, T, Ntw)
        ray_points = rng.rayleigh(1.0, Ntw)
        ray_points[[-1, 0]] = 0
        spline = sp.interpolate.CubicSpline(t_mod, ray_points)
        return spline(time_vector) / np.max(ray_points)

    def build(f_mod):
        rng = np.random.default_rng(seed)
        modulation = get_ray_modulation(f_mod, rng)
        nonstat = _add_am_modulation(carrier, modulation)
        return nonstat, modulation

    # ---------------------------------------------------------------- direct
    if f_mod is not None:
        if input_kurtosis is not None:
            print("f_mod is specified; input_kurtosis is ignored.")
        nonstat_signal, modulation = build(f_mod)
        return nonstat_signal, {
            "name": "Rayleigh amplitude modulation [-]",
            "modulation": [modulation],
            "params": {"f_mod": f_mod},
        }

    # -------------------------------------------------------------- optimize
    if input_kurtosis is None:
        raise ValueError("input_kurtosis is required when f_mod is not specified.")
    print("No modulation frequency given. Optimizing f_mod for given kurtosis.")
    bounds = {"f_mod": (max(2.0 / T, 0.5), fs / 4)}
    if param_bounds:
        bounds.update(param_bounds)

    def signed_err(f_mod_val):
        rng = np.random.default_rng(seed)
        modulation = get_ray_modulation(float(np.atleast_1d(f_mod_val)[0]), rng)
        return sp.stats.kurtosis(_add_am_modulation(carrier, modulation)) + 3 - input_kurtosis

    lo, hi = bounds["f_mod"]
    e_lo, e_hi = signed_err(lo), signed_err(hi)
    if e_lo * e_hi <= 0:
        f_mod = sp.optimize.brentq(signed_err, lo, hi, xtol=1e-2)
    else:  # target outside reachable range -> closest endpoint
        f_mod = lo if abs(e_lo) < abs(e_hi) else hi

    nonstat_signal, modulation = build(f_mod)

    achieved = sp.stats.kurtosis(nonstat_signal) + 3
    if abs(achieved - input_kurtosis) > 0.5:
        print(
            f"Target kurtosis {input_kurtosis} not reached (got {achieved:.2f}). "
            "Widen the f_mod bounds, or use a modulation with an extra shape "
            "parameter (e.g. the Beta modulation) for higher kurtosis."
        )
    return nonstat_signal, {
        "name": "Rayleigh amplitude modulation [-]",
        "modulation": [modulation],
        "params": {"f_mod": f_mod},
    }

def get_trapp_amplitude_modulation(
    carrier: np.ndarray,
    T: float,
    input_kurtosis: float = None,
    p: float = None,
    delta_m: float = None,
    seed: int = None,
    param_bounds: dict = None,
):
    """
    Apply amplitude modulation using a transformed power-law Rayleigh modulation.

    Two modes of operation
    ----------------------
        Direct
            If ``p`` and ``delta_m`` are both provided, the modulation function is
            built straight away and ``input_kurtosis`` is ignored.
        Optimize
            If either ``p`` or ``delta_m`` is left as ``None``, the missing one(s)
            are fitted so the output kurtosis matches ``input_kurtosis``. Provided
            parameters are held fixed.

    Parameters
    ----------
        carrier : np.ndarray
            Input Gaussian signal.
        T : float
            Signal duration in seconds.
        input_kurtosis : float, optional
            Target kurtosis. Required whenever at least one parameter is missing;
            ignored in direct mode.
        p : float, optional
            Exponent of the power-law transformation applied to the Gaussian
            modulation seed. Large p -> rare, spiky bursts -> high kurtosis.
        delta_m : float, optional
            Offset added to the modulation. Large delta_m -> the modulation tends
            to a constant -> kurtosis returns towards the Gaussian value.
        seed : int, optional
            Seed for reproducibility. If None a random seed is drawn once and used
            consistently, so the fitted parameters and the returned modulation always
            describe the same realization.
        param_bounds : dict, optional
            Override default optimization bounds, e.g. ``{'p': (0.1, 20)}``.

    Returns
    -------
        nonstat_signal : np.ndarray
            Non-stationary resulting signal.
        modulation : dict
            ``{'name', 'modulation', 'params'}`` where ``params`` reports the
            p/delta_m actually used.

    Source
    ------
    A. Trapp, M. J. Makua, and P. Wolfsteiner, "Fatigue assessment of
    amplitude-modulated non-stationary random vibration loading", Procedia Struct.
    Integr., vol. 17, pp. 379-386, 2019, doi: 10.1016/j.prostr.2019.08.050.
    """
    N = len(carrier)
    fs = N / T

    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31 - 1))

    # Low-frequency Gaussian seed of the modulation: it depends only on `seed`,
    # so every candidate (p, delta_m) is evaluated on the same realization.
    fpsd = np.array([0, 0.1, 0.2, 0.3, 0.4, fs / 2])
    psd = np.array([0, 0, 1, 1, 0, 0]) * 2 / (fpsd[4] + fpsd[3] - fpsd[2] - fpsd[1])
    _, gauss_mod_seed = get_stationary_gaussian(fpsd, psd, T, seed=seed)
    gauss_mod_seed = sp.signal.windows.tukey(len(gauss_mod_seed), alpha=0.01) * gauss_mod_seed

    def get_trapp_modulation(p, delta_m):
        return np.abs(gauss_mod_seed) ** p + delta_m

    def build(p, delta_m):
        modulation = get_trapp_modulation(p, delta_m)
        nonstat = _add_am_modulation(carrier, modulation)
        return nonstat, modulation

    given = {"p": p, "delta_m": delta_m}
    free = [k for k, v in given.items() if v is None]

    # ---------------------------------------------------------------- direct
    if not free:
        if input_kurtosis is not None:
            print("p and delta_m are both specified; input_kurtosis is ignored.")
        nonstat_signal, modulation = build(p, delta_m)
        return nonstat_signal, {
            "name": "Gaussian amplitude modulation [-]",
            "modulation": [modulation],
            "params": {"p": p, "delta_m": delta_m},
        }

    # -------------------------------------------------------------- optimize
    if input_kurtosis is None:
        raise ValueError(
            "input_kurtosis is required when p or delta_m is not specified."
        )
    print("No power-law parameters given. Optimizing parameters for given kurtosis.")
    bounds = {
        "p": (0.1, 100.0),
        "delta_m": (0.0, 10.0),
    }
    if param_bounds:
        bounds.update(param_bounds)

    def assemble(free_vals):
        q = dict(given)
        for k, v in zip(free, free_vals):
            q[k] = v
        return q["p"], q["delta_m"]

    def signed_err(free_vals):
        pp, dm = assemble(np.atleast_1d(free_vals))
        nonstat = _add_am_modulation(carrier, get_trapp_modulation(pp, dm))
        return sp.stats.kurtosis(nonstat) + 3 - input_kurtosis

    def abs_err(free_vals):
        pp, dm = assemble(np.atleast_1d(free_vals))
        nonstat = _add_am_modulation(carrier, get_trapp_modulation(pp, dm))
        kurt = sp.stats.kurtosis(nonstat) + 3
        skew = sp.stats.skew(nonstat)
        return abs(kurt - input_kurtosis) + abs(skew)

    free_bounds = [bounds[k] for k in free]

    if len(free) == 1:
        lo, hi = free_bounds[0]
        e_lo, e_hi = signed_err(lo), signed_err(hi)
        if e_lo * e_hi <= 0:
            best = [sp.optimize.brentq(signed_err, lo, hi, xtol=1e-2)]
        else:  # target outside reachable range -> closest endpoint
            best = [lo if abs(e_lo) < abs(e_hi) else hi]
    else:
        res = sp.optimize.differential_evolution(
            abs_err, free_bounds,
            seed=seed, tol=1e-2, polish=False, maxiter=200,
        )
        best = res.x

    p, delta_m = assemble(np.atleast_1d(best))
    nonstat_signal, modulation = build(p, delta_m)

    achieved = sp.stats.kurtosis(nonstat_signal) + 3
    if abs(achieved - input_kurtosis) > 0.5:
        print(
            f"Target kurtosis {input_kurtosis} not reached (got {achieved:.2f}). "
            "Free another parameter, or raise the p bound / lower delta_m for higher kurtosis."
        )
    return nonstat_signal, {
        "name": "Gaussian amplitude modulation [-]",
        "modulation": [modulation],
        "params": {"p": p, "delta_m": delta_m},
    }

def get_frequency_modulation(
    Sx:np.ndarray,
    SFT: sp.signal.ShortTimeFFT, 
    modulation_function:np.ndarray
    ):
    """
    Apply frequency shift modulation to a spectrogram Sx using a given modulation function.

    Parameters
    ---------
        Sx : np.ndarray 
            Input spectrogram (complex-valued).
        SFT : sp.signal.ShortTimeFFT
            Object with STFT/ISTFT functionality and attributes `delta_t`, `f`.
        modulation_function : array-like
            Frequency shift values over time.

    Returns
    -------
        nonstat_signal : np.ndarray
            Non stationary resulting signal
        modulation : dict
            Dictionary containing "name" of the modulation function and "modulation" time history 
    Source
    ------
    M. Clerc e S. Mallat, «Estimating deformations of stationary processes», Ann. Stat., vol. 31, fasc. 6, dic. 2003, doi: 10.1214/aos/1074290327.
    """
    if not isinstance(Sx,np.ndarray): Sx = np.asarray(Sx) 
    base_fft_window = np.pad(np.mean(np.abs(Sx), axis=1),(np.size(Sx,0), np.size(Sx,0)))
    time_bins = np.size(Sx,axis = 1)
    T = (time_bins-1)*SFT.delta_t
    original_times = np.linspace(0,T, len(modulation_function))
    sft_times = np.arange(0,time_bins) * SFT.delta_t
    df_sft = np.mean(np.diff(SFT.f))
    sft_modulation = np.interp(sft_times, original_times, modulation_function)/df_sft
    Sx_angle = np.angle(Sx)
    Sx_angle =np.random.uniform(-np.pi, np.pi, (np.size(Sx,0), np.size(Sx,1)))
    Sx_mag_modulated = np.zeros((np.size(Sx,0), np.size(Sx,1)))
    for  i, nf in enumerate(sft_modulation):
        Sx_mag_modulated[:,i] = np.roll(base_fft_window, round(nf))[np.size(Sx,0):2*np.size(Sx,0)]              
    Sx_modulated = Sx_mag_modulated*np.exp(1j * Sx_angle)
    nonstat_sign = SFT.istft(Sx_modulated)
    modulation_function = np.interp(
        np.linspace(0,1,len(nonstat_sign)),
        np.linspace(0,1,len(modulation_function)), 
        modulation_function)
    return nonstat_sign, {'name': 'Frequency shift [Hz]', 'modulation': [modulation_function]}


class NonStationaryNonGaussian(SingleChanSignal):
    """
    Generate a non-stationary, non-Gaussian signal based on a given PSD.
    NonStationaryNonGaussian is a child class of SingleChanSignal.
    
    Parameters:
        fpsd : np.ndarray 
            Frequency vector for the input PSD.
        psd : np.ndarray
            Power spectral density values.
        T : float
            Total duration of the signal [s].
        fs : float, optional 
            Wanted sampling frequency. If None -> 2*fpsd[-1]
        dfpsd : float, optional 
            Frequency resolution [Hz]. Default is 0.5.
        method : str, optional 
            Modulation method: 'beta_am', 'ray_am', 'trapp_am', or 'fm'. Default is 'beta_am'.
        params : dict, optional 
            Dictionary of parameters for the modulation method.
        name : str, optional 
            Name of the signal.
        var str, optional 
            Variable symbol, e.g., 'x'.
        unit : str, optional
            Signal unit. Default is '$m/s^2$'.
        seed_carr : int, optional 
            Random seed for reproducibility of the carrier.
        seed_mod : int, optional 
            Random seed for reproducibility of the modulating function.
        interp : str, optional 
            PSD interpolation method: 'lin' or 'log'. Default is 'lin'.
    """
    def __init__(
        self,
        fpsd:np.ndarray,
        psd:np.ndarray,
        T:float,
        fs:float = None,
        dfpsd:float = 0.5,
        method:str = 'beta_am',
        params = None,
        name:str="",
        var:str='x',
        unit:str="$m/s^2$",
        seed_carr:int=None,
        seed_mod:int=None,
        interp:str ='lin',
        quiet:bool = False):
        """
        Initializes a NonStationaryNonGaussian signal generator.

        Parameters
        ----------
            fpsd : np.ndarray
                Frequency vector for the input PSD.
            psd : np.ndarray
                Power spectral density values.
            T : float
                Total duration of the signal [s].
            fs : float, optional
                Wanted sampling frequency. If None -> 2*fpsd[-1].
            dfpsd : float, optional
                Frequency resolution [Hz]. Default is 0.5.
            method : str, optional
                Modulation method: 'beta_am', 'ray_am', 'trapp_am', or 'fm'. Default is 'beta_am'.
            params : dict, optional
                Dictionary of parameters for the modulation method.
            name : str, optional
                Name of the signal.
            var : str, optional
                Variable symbol, e.g., 'x'.
            unit : str, optional
                Signal unit. Default is '$m/s^2$'.
            seed_carr : int, optional
                Random seed for the carrier for reproducibility.
            seed_carr : int, optional
                Random seed for the modulation for reproducibility.
            interp : str, optional
                PSD interpolation method: 'lin' or 'log'. Default is 'lin'.
            quiet : bool, optional
                Suppresses the print of stats and initialization info. Default is False 
        """        
        
        method = method.lower()
        interp = interp.lower()
        interps = {'lin': lin_interp_psd,
                   'log': log_interp_psd}
        
        method_existance_check(interp, interps)
        
        if fs is None: 
            self.fs = fpsd[-1] * 2  # Nyquist frequency
        else:
            self.fs = fs
        
        if len(fpsd)!=len(psd):
            raise AttributeError('The frequency vector and the PSD vector are not the same size.')
        self.dfpsd = dfpsd
        self.fpsd,self.psd = interps[interp](fpsd, psd, n_points = int(fpsd[-1]/dfpsd), fs = self.fs) 
        self.T = T
        self.x, self.modulation_info = self._get_nonstationary_timehistory(method = method, params = params, seed_gauss = seed_carr, seed_mod = seed_mod)
        self.modulation = self.modulation_info['modulation']
        if 'params' in self.modulation_info.keys():
            self.mod_params = self.modulation_info['params']
        else:
            self.mod_params = []
        self.signal_type = f'Nonstationary - NonGaussian ({method})'
        super().__init__(
            x = self.x, 
            fs = self.fs, 
            dfpsd = dfpsd, 
            name = name, 
            var = var, 
            unit = unit, 
            signal_type=self.signal_type,
            quiet = quiet
            )
    
    
    def _get_gaussian_timehistory(self, seed = None):
        """
        Internal function called by __init__
        Generate a stationary Gaussian signal from the given PSD.

        Returns:
            signal : np.ndarray 
                The generated Gaussian signal.
        """
        _, signal = get_stationary_gaussian(self.fpsd, self.psd,self.T, seed)
        return signal
    
    def _get_nonstationary_timehistory(self, params, method = 'beta_am', seed_gauss = None, seed_mod = None):
        """
        Internal function called by __init__
        Designes a non-stationary process based off the parameters and methods imposed by the user.

        Parameters:
            params : dict
                Parameters for the modulation method.
            method : str, optional 
                Modulation method.
            seed_gauss : int, optional 
                Seed for Gaussian signal.
            seed_mod : int, optional
                Seed for modulation function.
    
        Returns:
            tuple: Non-stationary signal and modulation function info.
        """
        stationary_gaussian_signal = self._get_gaussian_timehistory(seed = seed_gauss)
        methods = {'beta_am': get_beta_amplitude_modulation,
                   'ray_am': get_rayleigh_amplitude_modulation,
                   'trapp_am': get_trapp_amplitude_modulation,
                   'fm' : get_frequency_modulation,
           }
        method_existance_check(method, methods)
        print(f'Estimating non-stationary signal from given parameters with "{method}" method')
        
        if method in ['fm']:
            nperseg = round(self.fs/2/self.dfpsd)
            self.x = stationary_gaussian_signal
            Sx, SFT = self.get_sftf(nperseg = nperseg, hop = nperseg//2, nargout=2)
            nonstationary_signal, modulation = methods[method](Sx, SFT, modulation_function = params)  
        else:
            nonstationary_signal, modulation = methods[method](stationary_gaussian_signal, self.T, **params, seed=seed_mod)  
        
        output_skewness = sp.stats.skew(nonstationary_signal)
        output_kurtosis = sp.stats.kurtosis(nonstationary_signal)+3
        print(f'Non-stationary signal generated with "{method}" method: skew = {output_skewness:.1f}, kurt = {output_kurtosis:.1f}')
        return nonstationary_signal, modulation
    
    def plot_modulation(self, ax = None, ylims = None, xlims = None):
        """
        Plot the modulation function used to transform the Gaussian signal.
        Parameters
        ----------
            ax : matplotlib.axes.Axes, optional
                Axis to plot on. Creates a new one if None.
            ylims : tuple, optional 
                y-axis limits.
            xlims : tuple, optional 
                x-axis limits.
        
        Returns
        -------
            ax : matplotlib.axes.Axes
                The axes with the plotted modulation function.
        """
        print(f'Plotting Timehistory')
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        color_list = ['k']+[f'C{i:d}' for i in range(len(self.modulation))]
        for i, band_modulation in enumerate(self.modulation):
            ax.plot(self.t,band_modulation,color_list[i], label = f"Modulation n.{i+1}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(self.modulation_info['name'])  
        ax.legend() 
        ax.grid(True,which = 'both')
        ax.minorticks_on()   
        ax.set_title(f'Modulation function {self.name}')
        if xlims is not None:
            ax.set_xlim(xlims)
        if ylims is not None:
            ax.set_ylim(ylims)
        return ax
        

if __name__ == "__main__":
    pass
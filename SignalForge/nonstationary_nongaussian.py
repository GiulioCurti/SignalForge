import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import *
from single_chan_signal import SingleChanSignal

'''
supporting functions / 'static methods' (no need of self object)
- get_beta_amplitude_modulation:            adds amplitude modulated non-stationarity to gaussian signal via beta distribution
- get_rayleigh_amplitude_modulation:        adds amplitude modulated non-stationarity to gaussian signal via rayleigh distribution
- get_trapp_amplitude_modulation:           adds amplitude modulated non-stationarity to gaussian signal via gaussian distribution
- get_frequency_modulation:                 designes frequency modulate non-stationary process from given carrier function and starting psd
'''

def _add_am_carrier(noise:np.ndarray, carrier:np.ndarray):
    """
    private function for adding amplitude modulation carrier to gaussian noise standardizing rms to the noise 
    """
    nonstat_sign = noise*carrier
    rms_correction = np.std(noise) / np.std(nonstat_sign)
    nonstat_sign = nonstat_sign * rms_correction
    return nonstat_sign

def _oneband_am_opt_fun(
    params:list, 
    gauss_signal:np.ndarray, 
    time_vector:np.ndarray, 
    carrier_fun:np.ndarray, 
    input_kurtosis:float, 
    seed:int
    ):
    """
    optimization function for oneband ampiltude modulated non stationary signals
    """
    carrier = carrier_fun(*params, time_vector, seed)
    nonstat_sign = _add_am_carrier(gauss_signal, carrier)
    transformed_kurtosis = sp.stats.kurtosis(nonstat_sign) + 3
    return np.abs(transformed_kurtosis - input_kurtosis)

def _get_banded_am_opt(
    gauss_signal:np.ndarray, 
    T:float, 
    input_kurtosis:float, 
    flims:list, 
    amplitude_modulation_fun:np.ndarray, 
    seed:int, 
    opt_settings:dict
    ):
    """
    optimization function for multiband ampiltude modulated non stationary signals
    """
    time_vector = np.linspace(0,1,len(gauss_signal))*T
    fs = T/len(gauss_signal)
    Nfw = len(input_kurtosis)
    flims = np.linspace(flims[0],flims[1],Nfw+1)
    fft_os = np.fft.rfft(gauss_signal)/fs
    fvec = np.fft.rfftfreq(len(gauss_signal), fs)
    nonstat_sign = np.zeros(len(gauss_signal))
    carrier = np.zeros((Nfw,len(gauss_signal)))
    for i in tqdm(range(Nfw)):
        Xos_win = np.zeros(len(fft_os), dtype=np.complex_)
        fwindow = (fvec >= flims[i]) & (fvec < flims[i+1])
        Xos_win[fwindow] = fft_os[fwindow]
        banded_signal = np.fft.irfft(Xos_win*fs)
        optim_res = sp.optimize.minimize(_oneband_am_opt_fun, args = (banded_signal, time_vector, amplitude_modulation_fun, input_kurtosis[i], seed),**opt_settings)
        carrier[i,:] = amplitude_modulation_fun(*optim_res.x, time_vector, seed)
        nonstat_sign += _add_am_carrier(banded_signal, carrier[i,:])
    return nonstat_sign, carrier

def get_beta_amplitude_modulation(
    gauss_signal: np.ndarray, 
    T: float, 
    input_kurtosis: float, 
    flims: list = None, 
    seed : int = None
    ):
    """
    Apply beta-distributed amplitude modulation to a Gaussian signal.

    Parameters
    ----------
        gauss_signal (array-like): Input Gaussian signal.
        T (float): Signal duration in seconds.
        input_kurtosis (float or array-like): Desired kurtosis of the output signal.
        flims (tuple, optional): Frequency limits for banded modulation.
        seed (int, optional): Seed for random number generation.

    Returns
    -------
        tuple: (Modulated signal, carrier information dictionary)
    
    Source
    ------
    D. Smallwood, «Vibration with Non-Gaussian Noise», J. IEST, vol. 52, fasc. 2, pp. 13–30, ott. 2009, doi: 10.17764/jiet.52.2.gh0444564n8765k1.
    """
    def get_beta_carrier(alpha, Ntw, time_vector, seed = None):
        """
        Generate a beta-distributed amplitude modulation carrier.

        Parameters:
            alpha (float): Shape parameter for the beta distribution.
            Ntw (int): Number of time windows.
            time_vector (array-like): Time axis for the signal.
            seed (int, optional): Seed for reproducibility.

        Returns:
            array-like: Normalized amplitude modulation carrier.
        """   
        beta = alpha
        t_mod = np.linspace(0,max(time_vector), round(Ntw))
        if seed: np.random.seed(seed=seed)
        beta_points = np.random.beta(alpha, beta, round(Ntw))
        beta_points  = 2*beta_points 
        beta_points[[-1,0]] = 0
        np.random.seed(seed=None)
        spline_interpolator = sp.interpolate.CubicSpline(t_mod, beta_points)
        beta_carrier = spline_interpolator(time_vector)/np.max(beta_points)   
        return beta_carrier
    if np.isscalar(input_kurtosis):
        time_vector = np.linspace(0,1,len(gauss_signal))*T
        optim_res = sp.optimize.minimize(
            _oneband_am_opt_fun, 
            args = (gauss_signal, time_vector, get_beta_carrier, input_kurtosis, seed), 
            x0=[1,40], 
            bounds=[(0.1,5), (30,len(gauss_signal)//2)]
            )
        carrier = get_beta_carrier(*optim_res.x, time_vector, seed)
        nonstat_sign = _add_am_carrier(gauss_signal, carrier)
        carrier = [carrier]
    else:
        opt_settings = {'x0':[1,40], 'bounds':[(0.5,5), (30,len(gauss_signal)//2)]}
        nonstat_sign, carrier = _get_banded_am_opt(
            gauss_signal, 
            T, 
            input_kurtosis, 
            flims, 
            get_beta_carrier, 
            seed, 
            opt_settings)
    return nonstat_sign, {'name': 'Beta amplitude modulation [-]', 'carrier': carrier}

def get_rayleigh_amplitude_modulation(
    gauss_signal: np.ndarray, 
    T: float, 
    input_kurtosis: float, 
    flims: list = None, 
    seed: int = None
    ):
    """
    Apply Rayleigh-distributed amplitude modulation to a Gaussian signal.

    Parameters
    ----------
        gauss_signal (array-like): Input Gaussian signal.
        T (float): Signal duration in seconds.
        input_kurtosis (float or array-like): Desired kurtosis of the output signal.
        flims (tuple, optional): Frequency limits for banded modulation.
        seed (int, optional): Seed for random number generation.

    Returns
    -------
        tuple: (Modulated signal, carrier information dictionary)
        
    Source
    ------
    D. Smallwood, «Vibration with Non-Gaussian Noise», J. IEST, vol. 52, fasc. 2, pp. 13–30, ott. 2009, doi: 10.17764/jiet.52.2.gh0444564n8765k1.
    """   
    def get_ray_carrier(sigma, Ntw, time_vector, seed = None):   
        """
        Generate a Rayleigh-distributed amplitude modulation carrier.

        Parameters:
            sigma (float): Scale parameter of the Rayleigh distribution.
            Ntw (int): Number of time windows.
            time_vector (array-like): Time axis for the signal.
            seed (int, optional): Seed for reproducibility.

        Returns:
            array-like: Normalized amplitude modulation carrier.
        """
        t_mod = np.linspace(0,max(time_vector), round(Ntw))
        if seed: np.random.seed(seed=seed)
        ray_points = np.random.rayleigh(sigma, round(Ntw))
        ray_points[[-1,0]] = 0
        np.random.seed(seed=None)
        spline_interpolator = sp.interpolate.CubicSpline(t_mod, ray_points)
        ray_carrier = spline_interpolator(time_vector)/np.max(ray_points)   
        return ray_carrier
    
    if np.isscalar(input_kurtosis):
        time_vector = np.linspace(0,1,len(gauss_signal))*T
        optim_res = sp.optimize.minimize(
            _oneband_am_opt_fun, 
            args = (gauss_signal, time_vector, get_ray_carrier, input_kurtosis, seed),
            x0=[2,40], 
            bounds=[(0.1,10), (30,len(gauss_signal)//2)]
            )
        carrier = get_ray_carrier(*optim_res.x, time_vector, seed)
        nonstat_sign = _add_am_carrier(gauss_signal, carrier)
        carrier = [carrier]
    else:
        if flims is None: raise KeyError('Parameter "flims" must be defined to get a specified spectral kurtosis')
        opt_settings  = {'x0':[2,40], 'bounds':[(0.1,10), (30,len(gauss_signal)//2)]}
        nonstat_sign, carrier = _get_banded_am_opt(gauss_signal, T, input_kurtosis, flims, get_ray_carrier, seed, opt_settings)
    return nonstat_sign, {'name': 'Rayleigh amplitude modulation [-]', 'carrier': carrier}

def get_trapp_amplitude_modulation(
    gauss_signal: np.ndarray, 
    T: float, 
    input_kurtosis:float, 
    flims:list = None, 
    seed : int = None
    ): # TODO: ADD MULTIBAND MODULATION
    """
    Apply Gaussian-envelope-based amplitude modulation using a transformed power-law Rayleigh carrier.

    Parameters
    ----------
        gauss_signal (array-like): Input Gaussian signal.
        T (float): Duration in seconds.
        input_kurtosis (float): Desired signal kurtosis.
        flims (tuple, optional): Placeholder for banded modulation (not yet implemented).
        seed (int, optional): Random seed for reproducibility.

    Returns
    -------
        tuple: (Modulated signal, carrier information dictionary)
    
    Source
    ------
    A. Trapp, M. J. Makua, e P. Wolfsteiner, «Fatigue assessment of amplitude-modulated non-stationary random vibration loading», Procedia Struct. Integr., vol. 17, pp. 379–386, 2019, doi: 10.1016/j.prostr.2019.08.050.
    """
    def get_trapp_carrier(p, delta_m, gaussian_carier): 
        """Generate amplitude modulation carrier based on power-law transformation."""  
        ray_carrier = np.abs(gaussian_carier)**p+delta_m
        return ray_carrier
    
    def trapp_am_opt_fun(params, gauss_carrier, input_kurtosis):
        """Optimization objective: match output signal kurtosis and reduce skew."""
        carrier = get_trapp_carrier(*params, gaussian_carier=gauss_carrier)
        nonstat_sign = _add_am_carrier(gauss_signal, carrier)
        transformed_kurtosis = sp.stats.kurtosis(nonstat_sign) + 3
        transformed_skew = sp.stats.skew(nonstat_sign)
        return np.abs(transformed_kurtosis - input_kurtosis)+np.abs(transformed_skew)
    
    if flims: 
        raise UserWarning('Multiband amplitude modulation not implemented yet for this method.')
    if seed: np.random.seed(seed=seed)
    fs = len(gauss_signal)/T
    fpsd = np.array([0,.2,.3,.4,.5,fs/2])
    psd = np.array([0,0,1,1,0,0])*2/(fpsd[4]+fpsd[3]-fpsd[2]-fpsd[1])
    _,gauss_carrier = get_stationary_gaussian(fpsd, psd, T, seed = None)
    gauss_carrier = sp.signal.windows.tukey(len(gauss_carrier),alpha = 0.01)*gauss_carrier
    
    if np.isscalar(input_kurtosis):
        optim_res = sp.optimize.minimize(
            trapp_am_opt_fun, 
            args = (gauss_carrier, input_kurtosis), 
            x0=[2,0], 
            bounds=[(0.1,100), (0,10)]
            )
        carrier = get_trapp_carrier(*optim_res.x, gaussian_carier = gauss_carrier)
        nonstat_sign = _add_am_carrier(gauss_signal, carrier)
        carrier = [carrier]
    # else:
    #     if flims is None: raise KeyError('Parameter "flims" must be defined to get a specified spectral kurtosis')
    #     opt_settings  = {'x0':[2,0], 'bounds':[(0.1,100), (0,10)]}
    #     nonstat_sign = _get_banded_am_opt(gauss_signal, T, input_kurtosis, flims, get_trapp_carrier, seed, opt_settings)
    
    return nonstat_sign, {'name': 'Gaussian amplitude modulation [-]', 'carrier': carrier}

def get_frequency_modulation(
    Sx:np.ndarray,
    SFT: sp.signal.ShortTimeFFT, 
    modulation_function:np.ndarray
    ):
    """
    Apply frequency shift modulation to a spectrogram Sx using a given modulation function.

    Parameters
    ---------
        Sx (np.ndarray): Input spectrogram (complex-valued).
        SFT: Object with STFT/ISTFT functionality and attributes `delta_t`, `f`.
        modulation_function (array-like): Frequency shift values over time.

    Returns
    -------
        tuple: (Time-domain modulated signal, carrier information dictionary)
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
    carrier = np.interp(
        np.linspace(0,1,len(nonstat_sign)),
        np.linspace(0,1,len(modulation_function)), 
        modulation_function)
    return nonstat_sign, {'name': 'Frequency shift [Hz]', 'carrier': [carrier]}


class NonStationaryNonGaussian(SingleChanSignal):
    """
    Generate a non-stationary, non-Gaussian signal based on a given PSD.
    NonStationaryNonGaussian is a child class of SingleChanSignal.
    
    Parameters:
        fpsd (np.ndarray): Frequency vector for the input PSD.
        psd (np.ndarray): Power spectral density values.
        T (float): Total duration of the signal [s].
        dfpsd (float, optional): Frequency resolution [Hz]. Default is 0.5.
        method (str, optional): Modulation method: 'beta_am', 'ray_am', 'trapp_am', or 'fm'. Default is 'beta_am'.
        params (dict, optional): Dictionary of parameters for the modulation method.
        name (str, optional): Name of the signal.
        var (str, optional): Variable symbol, e.g., 'x'.
        unit (str, optional): Signal unit. Default is '$m/s^2$'.
        seed (int, optional): Random seed for reproducibility.
        interp (str, optional): PSD interpolation method: 'lin' or 'log'. Default is 'lin'.
    """
    def __init__(
        self, 
        fpsd:np.ndarray, 
        psd:np.ndarray, 
        T:float, 
        dfpsd:float = 0.5, 
        method:str = 'beta_am', 
        params = None, 
        name:str="", 
        var:str='x', 
        unit:str="$m/s^2$", 
        seed:int=None, 
        interp:str ='lin'):        
        
        method = method.lower()
        interp = interp.lower()
        interps = {'lin': lin_interp_psd,
                   'log': log_interp_psd}
        
        method_existance_check(interp, interps)
        
        if len(fpsd)!=len(psd):
            raise AttributeError('The frequency vector and the PSD vector are not the same size.')
        self.dfpsd = dfpsd
        self.fpsd,self.psd = interps[interp](fpsd, psd, n_points = int(fpsd[-1]/dfpsd)) 
        self.fs = 2*self.fpsd[-1]
        self.T = T
        self.x, self.carrier = self._get_nonstationary_timehistory(method = method, params = params, seed_gauss = seed)
        self.signal_type = f'Nonstationary - NonGaussian ({method})'
        super().__init__(
            x = self.x, 
            fs = self.fs, 
            dfpsd = dfpsd, 
            name = name, 
            var = var, 
            unit = unit, 
            signal_type=self.signal_type
            )
    
    
    def _get_gaussian_timehistory(self, seed = None):
        """
        Internal function called by __init__
        Generate a stationary Gaussian signal from the given PSD.

        Returns:
            np.ndarray: The generated Gaussian signal.
        """
        _, signal = get_stationary_gaussian(self.fpsd, self.psd,self.T, seed)
        return signal
    
    def _get_nonstationary_timehistory(self, params, method = 'beta_am', seed_gauss = None, seed_mod = None):
        """
        Internal function called by __init__
        Designes a non-stationary process based off the parameters and methods imposed by the user.

        Parameters:
            params (dict): Parameters for the modulation method.
            method (str, optional): Modulation method.
            seed_gauss (int, optional): Seed for Gaussian signal.
            seed_mod (int, optional): Seed for modulation carrier.
    
        Returns:
            tuple: Non-stationary signal and modulation carrier info.
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
            nonstationary_signal, carrier = methods[method](Sx, SFT, modulation_function = params)  
        else:
            nonstationary_signal, carrier = methods[method](stationary_gaussian_signal, self.T, **params, seed=seed_mod)  
        
        output_skewness = sp.stats.skew(nonstationary_signal)
        output_kurtosis = sp.stats.kurtosis(nonstationary_signal)+3
        print(f'Non-stationary signal generated with "{method}" method: skew = {output_skewness:.1f}, kurt = {output_kurtosis:.1f}')
        return nonstationary_signal, carrier
    
    def plot_carrier(self, ax = None, ylims = None, xlims = None):
        """
        Plot the modulation carrier used to transform the Gaussian signal.
        Parameters
        ----------
            ax (matplotlib.axes.Axes, optional): Axis to plot on. Creates a new one if None.
            ylims (tuple, optional): y-axis limits.
            xlims (tuple, optional): x-axis limits.
        
        Returns
        -------
        ax (matplotlib.axes.Axes): The axes with the plotted carrier.
        """
        print(f'Plotting Timehistory')
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        color_list = ['k']+[f'C{i:d}' for i in range(len(self.carrier['carrier']))]
        for i, band_carrier in enumerate(self.carrier['carrier']):
            ax.plot(self.t,band_carrier,color_list[i], label = f"Carrier n.{i+1}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(self.carrier['name'])  
        ax.legend() 
        ax.grid(True,which = 'both')
        ax.minorticks_on()   
        ax.set_title(f'Carrier function {self.name}')
        if xlims is not None:
            ax.set_xlim(xlims)
        if ylims is not None:
            ax.set_ylim(ylims)
        return ax
        

if __name__ == "__main__":
    pass
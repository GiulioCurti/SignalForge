import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from utils import *
from single_chan_signal import SingleChanSignal


class StationaryGaussian(SingleChanSignal):
    """
    Generates a stationary Gaussian signal with a specified power spectral density (PSD).
    This class inherits from SingleChanSignal.

    Parameters
    ----------
    fpsd : np.ndarray
        Frequency vector of the power spectral density.
    psd : np.ndarray
        Power spectral density values.
    T : float
        Duration of the signal (in seconds).
    dfpsd : float, optional
        Frequency discretization for PSD storage, by default 0.5.
    name : str, optional
        Name of the signal (for plots), by default "".
    var : str, optional
        Variable name (for plots), by default "x".
    unit : str, optional
        Unit of measurement of the signal (for plots), by default "$m/s^2$".
    seed : int, optional
        Seed for the random generator (for reproducibility), by default None.
    interp : str, optional
        Interpolation method for PSD ("lin" or "log"), by default "lin".
    """

    def __init__(
        self,
        fpsd: np.ndarray,
        psd: np.ndarray,
        T: float,
        dfpsd: float = 0.5,
        name: str = "",
        var: str = "x",
        unit: str = "$m/s^2$",
        seed: int = None,
        interp: str = "lin"
    ):
        interp = interp.lower()

        # Dictionary of available interpolation functions
        interps = {
            'lin': lin_interp_psd,
            'log': log_interp_psd
        }

        # Validate interpolation method
        method_existance_check(interp, interps)

        # Check input dimensions
        if len(fpsd) != len(psd):
            raise AttributeError(
                'The frequency vector and the PSD vector must be the same size.'
            )

        # Set internal parameters
        self.fs = 2 * fpsd[-1]  # Nyquist frequency
        self.T = T

        # Generate time-domain Gaussian signal
        self.x = self._get_timehistory(fpsd, psd, seed)

        self.signal_type = 'Stationary - Gaussian'

        # Initialize parent class with signal and metadata
        super().__init__(
            x=self.x,
            fs=self.fs,
            dfpsd=dfpsd,
            name=name,
            var=var,
            unit=unit,
            signal_type=self.signal_type
        )

    def _get_timehistory(
        self,
        fpsd: np.ndarray,
        psd: np.ndarray,
        seed: int = None
    ) -> np.ndarray:
        """
        Generates a Gaussian time history from the given PSD.
        -> called by __init__
        
        Parameters
        ----------
        fpsd : np.ndarray
            Frequency vector of the power spectral density.
        psd : np.ndarray
            Power spectral density values.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Time-domain Gaussian signal.
        """
        _, signal = get_stationary_gaussian(fpsd, psd, self.T, seed)
        return signal


if __name__ == "__main__":
    pass
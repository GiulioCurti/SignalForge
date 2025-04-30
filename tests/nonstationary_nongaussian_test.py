import numpy as np
import pytest
import scipy.stats as stats
from SignalForge.nonstationary_nongaussian import (
    NonStationaryNonGaussian,
    get_beta_amplitude_modulation,
    get_rayleigh_amplitude_modulation,
    get_trapp_amplitude_modulation,
    get_frequency_modulation,
)
from types import SimpleNamespace


@pytest.fixture
def sample_psd_data():
    fpsd = [10,15]
    psd = [1,1]
    T = 200.0  # 20 seconds
    fs = 200
    return fpsd, psd, T, fs

@pytest.fixture
def gaussian_signal():
    np.random.seed(42)
    return np.random.randn(10000)

def test_class_initialization_beta_am(sample_psd_data):
    fpsd, psd, T ,fs = sample_psd_data
    params = {'input_kurtosis': 6}
    sig = NonStationaryNonGaussian(fpsd, psd, T, fs=fs, method='beta_am', params=params, seed=42)
    assert isinstance(sig, NonStationaryNonGaussian)
    assert sig.x.shape[0] == int(sig.T * sig.fs)

def test_invalid_psd_lengths():
    fpsd = np.linspace(0, 500, 100)
    psd = np.ones(50)  # mismatched
    with pytest.raises(AttributeError):
        NonStationaryNonGaussian(fpsd, psd, T=2.0)

def test_beta_amplitude_modulation_output_shape(gaussian_signal):
    T = 2.0
    input_kurtosis = 6
    y, carrier = get_beta_amplitude_modulation(gaussian_signal, T, input_kurtosis)
    assert isinstance(y, np.ndarray)
    assert len(y) == len(gaussian_signal)
    assert "carrier" in carrier

def test_rayleigh_amplitude_modulation_output_shape(gaussian_signal):
    T = 2.0
    input_kurtosis = 6
    y, carrier = get_rayleigh_amplitude_modulation(gaussian_signal, T, input_kurtosis)
    assert isinstance(y, np.ndarray)
    assert len(y) == len(gaussian_signal)
    assert "carrier" in carrier

def test_trapp_amplitude_modulation_output_shape(gaussian_signal):
    T = 2.0
    input_kurtosis = 6
    y, carrier = get_trapp_amplitude_modulation(gaussian_signal, T, input_kurtosis)
    assert isinstance(y, np.ndarray)
    assert len(y) == len(gaussian_signal)
    assert "carrier" in carrier

def test_frequency_modulation_output_shape(sample_psd_data):
    fpsd, psd, T ,fs = sample_psd_data
    modulation_function = 10+10*np.sin(np.linspace(0,2*np.pi,100))
    sig = NonStationaryNonGaussian(fpsd, psd, T, fs=fs, method='fm', params=modulation_function, seed=42)
    assert isinstance(sig.x, np.ndarray)
    assert len(sig.x) > 0
    assert "carrier" in sig.carrier

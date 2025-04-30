import numpy as np
import pytest
from SignalForge.stationary_gaussian import *

@pytest.fixture
def sample_psd_data():
    fpsd = [10,20]
    psd = [1,1]
    T = 20.0  # 2 seconds
    fs = 200
    return fpsd, psd, T, fs

def test_initialization(sample_psd_data):
    fpsd, psd, T, fs = sample_psd_data
    sg = StationaryGaussian(fpsd, psd, T, fs=fs, seed=123456)

    assert isinstance(sg, StationaryGaussian)
    assert hasattr(sg, "x")
    assert sg.signal_type == "Stationary - Gaussian"
    assert sg.T == T
    assert sg.fs == fs
    assert len(sg.x) == int(T * sg.fs)

def test_interp_mode_validation(sample_psd_data):
    fpsd, psd, T, fs = sample_psd_data
    with pytest.raises(KeyError):
        StationaryGaussian(fpsd, psd, T, fs = fs, interp="invalid")

def test_frequency_and_psd_size_check():
    fpsd = np.linspace(0, 500, 100)
    psd = np.ones(50)  # Wrong size
    T = 1.0
    with pytest.raises(AttributeError):
        StationaryGaussian(fpsd, psd, T)

def test_time_domain_signal_properties(sample_psd_data):
    fpsd, psd, T, fs = sample_psd_data
    sg = StationaryGaussian(fpsd, psd, T, fs = fs,seed=42)
    
    # Check that signal has near-zero mean for Gaussian
    assert abs(np.mean(sg.x)) < 0.1

    # Check the RMS power is roughly consistent with the PSD area
    expected_power = np.trapz(psd, fpsd)
    actual_power = np.var(sg.x)
    assert np.isclose(actual_power, expected_power, rtol=0.01)  # loose tolerance for randomness

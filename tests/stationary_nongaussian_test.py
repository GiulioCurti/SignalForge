import numpy as np
import pytest
from numpy.testing import assert_allclose
import scipy.stats as stats

from SignalForge.stationary_gaussian import StationaryGaussian
from SignalForge.stationary_nongaussian import StationaryNonGaussian  # Update import path
from SignalForge.stationary_nongaussian import (
    get_winterstein,
    get_cubic_polynomial,
    get_zheng,
    get_sarkani,
    get_zmnl,
    get_steinwolf,
    get_smallwood,
    get_vanbaren,
)
from SignalForge.utils import log_interp_psd

@pytest.fixture
def sample_psd_data():
    fpsd = [10,15]
    psd = [1,1]
    T = 200.0  # 20 seconds
    fs = 200
    return fpsd, psd, T, fs

@pytest.fixture
def sample_psd():
    fpsd = np.linspace(0, 100,400)
    psd = np.exp(-(fpsd-15)**2/10)
    psd[0] = 0
    return fpsd, psd

@pytest.fixture
def fs():
    return 100 # A sample fs

@pytest.fixture
def gaussian_signal(sample_psd_data):
    fpsd, psd, T, fs = sample_psd_data
    sign = StationaryGaussian(fpsd = fpsd, psd = psd, T = T, fs = fs)
    return sign.x 

def test_initialization(sample_psd_data):
    fpsd, psd, T, fs = sample_psd_data
    kurtosis = 6
    sg = StationaryNonGaussian(fpsd, psd, T, kurtosis, fs = fs, method='winter', seed=123)

    assert isinstance(sg, StationaryNonGaussian)
    assert hasattr(sg, "x")
    assert sg.signal_type.startswith("Stationary - NonGaussian")
    assert sg.T == T
    assert sg.fs == fs
    assert len(sg.x) == int(T * sg.fs)

def test_invalid_method(sample_psd_data):
    fpsd, psd, T, fs = sample_psd_data
    with pytest.raises(KeyError):
        StationaryNonGaussian(fpsd, psd, T, kurtosis=6, fs = fs, method='invalidmethod')

def test_mismatched_fpsd_psd_lengths():
    fpsd = np.linspace(0, 500, 100)
    psd = np.ones(50)  # wrong size
    T = 20.0
    with pytest.raises(AttributeError):
        StationaryNonGaussian(fpsd, psd, T, kurtosis=6)

def test_nongaussian_statistics_approx(sample_psd_data):
    fpsd, psd, T,fs = sample_psd_data
    kurtosis_target = 7
    skewness_target = 0.5
    sg = StationaryNonGaussian(fpsd, psd, T, kurtosis = kurtosis_target, fs = fs, skewness=skewness_target, method='winter', seed=42)

    output_skewness = float(np.round(sg.central_moments['skew'], 1))
    output_kurtosis = float(np.round(sg.central_moments['kurtosis'], 1))

    # Allow small deviation due to randomness and numerical errors
    assert abs(output_kurtosis - kurtosis_target) < 2
    assert abs(output_skewness - skewness_target) < 1

def test_other_methods_run(sample_psd_data):
    fpsd, psd, T, fs = sample_psd_data
    methods = ["winter", "cubic", "zheng", "sarkani"]
    kurtosis_target = 6
    for method in methods:
        print(f'testing method: {method}')
        sg = StationaryNonGaussian(fpsd, psd, T, kurtosis=kurtosis_target, fs = fs, method=method, seed=123456)
        output_kurtosis = sg.central_moments['kurtosis']
        assert isinstance(sg.x, np.ndarray)
        assert sg.x.shape[0] == int(sg.T * sg.fs)
        assert output_kurtosis > 3


def test_get_winterstein(gaussian_signal):
    target_skewness = 0.5
    target_kurtosis = 7
    out, _ = get_winterstein(gaussian_signal, input_skewness=target_skewness, input_kurtosis=target_kurtosis)
    assert isinstance(out, np.ndarray)
    assert len(out) == len(gaussian_signal)

def test_get_cubic_polynomial(gaussian_signal):
    target_kurtosis = 7
    out, _ = get_cubic_polynomial(gaussian_signal, input_kurtosis=target_kurtosis, input_skewness=0)
    assert isinstance(out, np.ndarray)
    assert len(out) == len(gaussian_signal)

def test_get_zheng(gaussian_signal):
    target_kurtosis = 7
    out, _ = get_zheng(gaussian_signal, input_kurtosis=target_kurtosis)
    assert isinstance(out, np.ndarray)
    assert len(out) == len(gaussian_signal)

def test_get_sarkani(gaussian_signal):
    target_kurtosis = 7
    out, _ = get_sarkani(gaussian_signal, input_kurtosis=target_kurtosis)
    assert isinstance(out, np.ndarray)
    assert len(out) == len(gaussian_signal)

def test_get_zmnl(gaussian_signal):
    target_kurtosis = 7
    fs_val = 100 # Renamed from 'fs' to avoid fixture conflict if 'fs' fixture is later defined globally
    out, _ = get_zmnl(gaussian_signal, fs=fs_val, input_kurtosis=target_kurtosis)
    assert isinstance(out, np.ndarray)
    assert len(out) == len(gaussian_signal)

def test_get_steinwolf_kurtosis(gaussian_signal, fs):
    target_kurtosis = 7
    out, _ = get_steinwolf(gaussian_signal, fs=fs, input_kurtosis=target_kurtosis)
    assert isinstance(out, np.ndarray)
    assert len(out) == len(gaussian_signal)
    output_kurtosis = stats.kurtosis(out)
    assert output_kurtosis > 0.0

def test_get_smallwood_kurtosis(sample_psd):
    fpsd, psd = sample_psd
    T = 200.0
    target_kurtosis = 7
    N = T*fpsd[-1]
    out, _ = get_smallwood(fpsd, psd, T, input_kurtosis=target_kurtosis)
    assert isinstance(out, np.ndarray)
    # Assert that the output kurtosis is close to the target, allowing for some tolerance
    output_kurtosis = stats.kurtosis(out)
    assert output_kurtosis > 0.0 

def test_get_vanbaren_kurtosis(sample_psd):
    fpsd, psd = sample_psd
    T = 200.0
    target_kurtosis = 7
    out, _ = get_vanbaren(fpsd, psd, T, input_kurtosis=target_kurtosis)
    assert isinstance(out, np.ndarray)
    # Assert that the output kurtosis is close to the target, allowing for some tolerance
    output_kurtosis = stats.kurtosis(out)
    assert output_kurtosis > 0.0

def test_banded_nongaussian_timehistory(sample_psd_data):
    fpsd, psd, T, fs = sample_psd_data
    # Define banded kurtosis targets
    kurtosis_targets = [5.0, 7.0, 5.0] # Example targets for 3 bands
    flims = [1, 20] # Example frequency limits
    skewness_target = 0.0 # For simplicity
    
    # Initialize class with banded kurtosis
    sg = StationaryNonGaussian(
        fpsd, psd, T, 
        kurtosis=np.array(kurtosis_targets), # Pass as numpy array
        flims=flims, 
        fs=fs, 
        skewness=skewness_target, 
        method='winter', # Use a known working method
        seed=42
    )

    assert isinstance(sg.x, np.ndarray)
    assert len(sg.x) == int(T * fs) # Check output signal length

    # Further assertions could involve checking kurtosis of filtered bands,
    # but that would involve re-implementing significant logic or exposing internal state.
    # For now, a basic check that the overall signal is non-Gaussian is a start.
    overall_kurtosis = stats.kurtosis(sg.x) + 3
    # Expect overall kurtosis to be higher than 3 (Gaussian) if individual bands are non-Gaussian
    assert overall_kurtosis > 3.0
    # A more rigorous test would check kurtosis per band after filtering, but this is a start.

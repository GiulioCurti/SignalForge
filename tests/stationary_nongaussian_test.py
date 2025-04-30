import numpy as np
import pytest
from numpy.testing import assert_allclose
import scipy.stats as stats

from SignalForge.stationary_nongaussian import StationaryNonGaussian  # Update import path
from SignalForge.stationary_nongaussian import (
    get_winterstein,
    get_cubic_polinomial,
    get_zheng,
    get_sarkani,
    get_zmnl,
    get_steinwolf,
    get_smallwood,
    get_vanbaren,
)

@pytest.fixture
def sample_psd_data():
    fpsd = [10,15]
    psd = [1,1]
    T = 200.0  # 20 seconds
    fs = 200
    return fpsd, psd, T, fs

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
        assert abs(output_kurtosis - kurtosis_target) < 2

# @pytest.fixture
# def gaussian_signal():
#     np.random.seed(0)
#     return np.random.randn(10000)  # Large enough for good statistics

# @pytest.fixture
# def sample_psd():
#     fpsd = np.linspace(0, 500, 100)
#     psd = 0.1 * np.ones_like(fpsd)
#     return fpsd, psd

# @pytest.fixture
# def fs():
#     return 1000  # typical sampling rate

# def test_get_winterstein(gaussian_signal):
#     target_skewness = 0.5
#     target_kurtosis = 7
#     out, _ = get_winterstein(gaussian_signal, input_skewness=target_skewness, input_kurtosis=target_kurtosis)
#     assert isinstance(out, np.ndarray)
#     assert len(out) == len(gaussian_signal)
#     assert abs(stats.kurtosis(out) + 3 - target_kurtosis) < 2

# def test_get_cubic_polinomial(gaussian_signal):
#     target_kurtosis = 7
#     out, _ = get_cubic_polinomial(gaussian_signal, input_kurtosis=target_kurtosis, input_skewness=0)
#     assert isinstance(out, np.ndarray)
#     assert len(out) == len(gaussian_signal)
#     assert abs(stats.kurtosis(out) + 3 - target_kurtosis) < 2

# def test_get_zheng(gaussian_signal):
#     target_kurtosis = 7
#     out, _ = get_zheng(gaussian_signal, input_kurtosis=target_kurtosis)
#     assert isinstance(out, np.ndarray)
#     assert len(out) == len(gaussian_signal)
#     assert abs(stats.kurtosis(out) + 3 - target_kurtosis) < 2

# def test_get_sarkani(gaussian_signal):
#     target_kurtosis = 7
#     out, _ = get_sarkani(gaussian_signal, input_kurtosis=target_kurtosis)
#     assert isinstance(out, np.ndarray)
#     assert len(out) == len(gaussian_signal)
#     assert abs(stats.kurtosis(out) + 3 - target_kurtosis) < 2

# def test_get_zmnl(gaussian_signal, fs):
#     target_kurtosis = 7
#     out, _ = get_zmnl(gaussian_signal, fs=fs, input_kurtosis=target_kurtosis)
#     assert isinstance(out, np.ndarray)
#     assert len(out) == len(gaussian_signal)
#     assert abs(stats.kurtosis(out) + 3 - target_kurtosis) < 2

# @pytest.mark.xfail(reason="Steinwolf method flagged WIP in the code comments")
# def test_get_steinwolf(gaussian_signal, fs):
#     target_kurtosis = 7
#     out, _ = get_steinwolf(gaussian_signal, fs=fs, input_kurtosis=target_kurtosis)
#     assert isinstance(out, np.ndarray)
#     assert len(out) == len(gaussian_signal)
#     # Note: This test is expected to fail currently

# @pytest.mark.skip(reason="Smallwood method marked NOT WORKING in the comments")
# def test_get_smallwood(sample_psd):
#     fpsd, psd = sample_psd
#     T = 2.0
#     target_kurtosis = 7
#     out, _ = get_smallwood(fpsd, psd, T, input_kurtosis=target_kurtosis)
#     assert isinstance(out, np.ndarray)

# @pytest.mark.skip(reason="Vanbaren method marked unstable and verbose optimization")
# def test_get_vanbaren(sample_psd):
#     fpsd, psd = sample_psd
#     T = 2.0
#     target_kurtosis = 7
#     out, _ = get_vanbaren(fpsd, psd, T, input_kurtosis=target_kurtosis)
#     assert isinstance(out, np.ndarray)
import numpy as np
import pytest
from SignalForge.single_chan_signal import *
from numpy.testing import assert_allclose, assert_almost_equal

@pytest.fixture
def sample_signal():
    np.random.seed(42)
    fs = 1000  # Hz
    T = 1.0    # seconds
    N = int(T * fs)
    t = np.linspace(0, T, N, endpoint=False)
    x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.random.randn(N)
    return SingleChanSignal(x=x, fs=fs)

def test_initialization(sample_signal):
    assert isinstance(sample_signal, SingleChanSignal)
    assert sample_signal.N == len(sample_signal.x)
    assert sample_signal.T == sample_signal.N / sample_signal.fs

def test_time_vector(sample_signal):
    t = sample_signal.t
    assert np.allclose(np.mean(np.diff(t)), 1 / sample_signal.fs, rtol=1e-3)
    assert len(t) == sample_signal.N

def test_fft_properties(sample_signal):
    assert len(sample_signal.fft_ts) == sample_signal.N
    assert len(sample_signal.f_fft_ts) == sample_signal.N
    assert len(sample_signal.fft_os) == int(np.ceil((sample_signal.N + 1) / 2))

def test_psd(sample_signal):
    f, psd = sample_signal.get_psd()
    assert len(f) == len(psd)
    assert f[0] == 0

def test_central_moments(sample_signal):
    cm = sample_signal.central_moments
    assert "rms" in cm and cm["rms"] > 0
    assert "kurtosis" in cm and cm["kurtosis"] > 0

def test_spectral_moments(sample_signal):
    sm = sample_signal.spectral_moments
    for key in ['smom0', 'smom1', 'smom2', 'smom4', 'smom6']:
        assert key in sm

def test_pdf_shape(sample_signal):
    grid, prob = sample_signal.get_pdf()
    assert len(grid) == len(prob)

def test_bestfit_gaussian(sample_signal):
    grid, pgauss = sample_signal.get_bestfit_gaussian_pdf()
    assert len(grid) == len(pgauss)

def test_hilbert_output(sample_signal):
    analytic, amp, phase = sample_signal.get_hilbert()
    assert len(analytic) == len(sample_signal.x)
    assert len(amp) == len(sample_signal.x)
    assert len(phase) == len(sample_signal.x)

def test_add_scalar(sample_signal):
    new_sig = sample_signal + 3
    assert isinstance(new_sig, SingleChanSignal)
    assert_allclose(new_sig.x, sample_signal.x + 3)

# You can test plotting functions to make sure they run without exceptions
def test_plot_methods_run(sample_signal):
    sample_signal.plot()
    sample_signal.plot_pdf()
    sample_signal.plot_fft()
    sample_signal.plot_psd()
    sample_signal.plot_spectrogram()

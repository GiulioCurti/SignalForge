import numpy as np
import pytest
from SignalForge.utils import *

def test_method_existance_check_valid():
    methods = {'a': lambda x: x}
    method_existance_check('a', methods)  # Should not raise

def test_method_existance_check_invalid():
    methods = {'a': lambda x: x}
    with pytest.raises(KeyError):
        method_existance_check('b', methods)

def test_step_interp():
    xp = [0, 1, 2, 3]
    yp = [10, 20, 30, 40]
    x = [0.5, 1.5, 2.5]
    result = step_interp(x, xp, yp)
    assert np.allclose(result, [20, 30, 40])

def test_lin_interp_psd():
    fpsd = np.array([0,10,20,30,200])
    psd = np.array([0,0,1,0,0])
    var_exp = np.trapz(psd,fpsd)
    npoints = 1000
    new_fpsd, new_psd = lin_interp_psd(fpsd, psd, npoints)
    var_interp = np.trapz(new_psd,new_fpsd)
    assert len(new_fpsd) == npoints
    assert len(new_psd) == npoints
    assert new_psd[0] == 0
    assert np.all(new_psd >= 0)
    assert np.isclose(var_exp, var_interp, rtol=0.1)

def test_log_interp_psd():
    fpsd = np.array([10,20,30])
    psd = np.array([0,1,0])
    var_exp = 0.316
    npoints = 2000
    fs = 60
    new_fpsd, new_psd = log_interp_psd(fpsd, psd, npoints, fs)
    var_interp = np.trapz(new_psd,new_fpsd)
    assert len(new_fpsd) == npoints
    assert len(new_psd) == npoints
    assert new_psd[0] == 0
    assert np.all(new_psd >= 0)
    assert np.isclose(var_exp, var_interp, rtol=0.1)

def test_get_stat_mom():
    var = np.linspace(0, 1, 100)
    prob = np.ones_like(var)
    moment = get_stat_mom(var, prob, 1)
    assert 0.4 < moment < 0.6  # Should be near mean of linspace(0,1)

def test_alpha_spec_index():
    f = np.linspace(1, 100, 500)
    psd = np.ones_like(f)
    idx = alpha_spec_index(f, psd, order=1, deriv_order=1)
    assert isinstance(idx, float)

def test_get_psd_impulse_response():
    f = np.linspace(0, 100, 500)
    psd = np.ones_like(f)
    h = get_psd_impulse_response(f, psd, 1024)
    assert len(h) == 1024

def test_get_nonstat_index():
    signal = np.random.randn(1000)
    kurtosis_trace = get_nonstat_index(signal, winsize=100, idx_type='kurtosis')
    assert len(kurtosis_trace) == len(signal) - 100 + 1

def test_get_banded_spectral_kurtosis():
    signal = np.random.randn(1024)
    fvec, SK = get_banded_spectral_kurtosis(signal, dt=1/1000, n_bins=5)
    assert len(fvec) == 5
    assert len(SK) == 5

def test_get_welch_spectral_kurtosis():
    signal = np.random.randn(2048)
    f, SK = get_welch_spectral_kurtosis(signal, Nfft=256, noverlap=64)
    assert len(f) == len(SK)

def test_get_stationary_gaussian():
    f = [0,10,20,200]
    psd = [0,0,1,200]
    var = np.trapz(psd, f)
    _, signal = get_stationary_gaussian(f, psd, T=100.0, seed=42)
    signal_var = np.var(signal)
    assert np.isclose(var, signal_var, rtol=0.1)

def test_gaussian_bell():
    grid = np.linspace(-5, 5, 100)
    y = gaussian_bell(grid, var=1)
    assert np.all(y >= 0)
    assert np.isclose(np.trapz(y, grid), 1, rtol=0.1)  # Should approximately integrate to 1

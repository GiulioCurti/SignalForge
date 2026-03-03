import numpy as np
import pytest
from SignalForge.utils import *
import scipy.signal # Added for hilbert test

def get_signals():
    stat_signal = np.random.randn(10000)
    nonstat_signal = np.stack([np.random.randn(500),10*np.random.randn(500)])   
    return stat_signal, nonstat_signal.flatten()

@pytest.fixture
def sample_real_signal():
    np.random.seed(42)
    return np.sin(np.linspace(0, 10 * np.pi, 1000)) + 0.1 * np.random.randn(1000)

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
    var_exp = np.trapezoid(psd,fpsd)
    npoints = 1000
    new_fpsd, new_psd = lin_interp_psd(fpsd, psd, npoints)
    var_interp = np.trapezoid(new_psd,new_fpsd)
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
    var_interp = np.trapezoid(new_psd,new_fpsd)
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

def test_get_stat_history():
    stat_signal, _ = get_signals()
    kurtosis_trace = get_stat_history(stat_signal, winsize=100, idx_type='kurtosis')
    mean_trace = get_stat_history(stat_signal, winsize=100, idx_type='mean')
    rms_trace = get_stat_history(stat_signal, winsize=100, idx_type='rms')
    var_trace = get_stat_history(stat_signal, winsize=100, idx_type='var')

    mean_kurt = np.mean(kurtosis_trace)
    mean_mean = np.mean(mean_trace)
    mean_rms = np.mean(rms_trace)
    mean_var = np.mean(var_trace)
    assert np.isclose(3, mean_kurt, rtol=0.1)
    assert np.isclose(1, mean_mean+1, rtol=0.1)
    assert np.isclose(1, mean_rms, rtol=0.1)
    assert np.isclose(1, mean_var, rtol=0.1)

def test_get_nnst_index():
    stat_signal, nonstat_signal = get_signals()
    test_stat = get_nnst_index(stat_signal, nperseg=100)
    test_nonstat = get_nnst_index(nonstat_signal, nperseg=100)
    assert test_stat['outcome']=='Stationary'
    assert test_nonstat['outcome']=='Non-stationary'

def test_get_adf_index():
    stat_signal, _ = get_signals()
    test_stat = get_adf_index(stat_signal)
    assert isinstance(test_stat['outcome'],str)

def test_get_kpss_index():
    stat_signal, _ = get_signals()
    test_stat = get_kpss_index(stat_signal)
    assert isinstance(test_stat['outcome'],str)

def test_print_nonstat_results_runs():
    # This test just ensures the function runs without error with typical input
    test_results = {'outcome': 'Stationary', 'test': 0.1, 'p-value': 0.05, 'lag': 5, 'crit_values': {'1%': -3.5, '5%': -2.8}}
    try:
        print_nonstat_results(test_results)
        assert True # If it reaches here, it ran without error
    except Exception as e:
        pytest.fail(f"print_nonstat_results raised an exception: {e}")

def test_print_nonstat_results_runs():
    # This test just ensures the function runs without error with typical input
    test_results = {'outcome': 'Stationary', 'test': 0.1, 'p-value': 0.05, 'lag': 5, 'crit_values': {'1%': -3.5, '5%': -2.8}}
    try:
        print_nonstat_results(test_results)
        assert True # If it reaches here, it ran without error
    except Exception as e:
        pytest.fail(f"print_nonstat_results raised an exception: {e}")

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
    var = np.trapezoid(psd, f)
    _, signal = get_stationary_gaussian(f, psd, T=100.0, seed=42)
    signal_var = np.var(signal)
    assert np.isclose(var, signal_var, rtol=0.1)

def test_gaussian_bell():
    grid = np.linspace(-5, 5, 100)
    y = gaussian_bell(grid, var=1)
    assert np.all(y >= 0)
    assert np.isclose(np.trapezoid(y, grid), 1, rtol=0.1)  # Should approximately integrate to 1

def test_get_hilbert_output(sample_real_signal):
    analytic, amp, phase = get_hilbert(sample_real_signal)
    
    assert isinstance(analytic, np.ndarray)
    assert np.iscomplexobj(analytic)
    assert len(analytic) == len(sample_real_signal)
    
    assert isinstance(amp, np.ndarray)
    assert np.isrealobj(amp)
    assert len(amp) == len(sample_real_signal)
    
    assert isinstance(phase, np.ndarray)
    assert np.isrealobj(phase)
    assert len(phase) == len(sample_real_signal)
    
    # Basic check: amplitude envelope should be non-negative
    assert np.all(amp >= 0)
    
    # Check relationship: abs(analytic_signal) == amplitude_envelope
    assert np.allclose(np.abs(analytic), amp)

def test_perfect_passband_filter_output(sample_real_signal):
    # Test a simple passband filter
    # Original signal has components around DC to Nyquist.
    # Let's create a signal with known frequency content, e.g., a sum of sines
    fs = 100 # Hz
    t = np.arange(0, 10, 1/fs)
    signal = np.sin(2*np.pi*5*t) + np.sin(2*np.pi*20*t) + np.sin(2*np.pi*40*t) # Frequencies at 5, 20, 40 Hz
    
    # Filter for 10-30 Hz (normalized: 0.1-0.3)
    fl_n = 10/fs # 0.1
    fu_n = 30/fs # 0.3
    filtered_signal = perfect_passband_filter(signal, fl_n, fu_n)
    
    assert isinstance(filtered_signal, np.ndarray)
    assert len(filtered_signal) == len(signal)
    
    # Check that the filtered signal primarily contains the 20 Hz component
    # A simple way is to check the power of original vs filtered signal in the band
    # This might require FFT and checking magnitudes, which is more involved for a quick test.
    # For now, check that the filtered signal is not zero and is different from original
    assert not np.allclose(filtered_signal, signal, atol=1e-5)
    assert np.sum(np.abs(filtered_signal)) > 0
    
    # More rigorous check would involve FFT of filtered signal
    fft_filtered = np.abs(np.fft.rfft(filtered_signal))
    freqs = np.fft.rfftfreq(len(filtered_signal), 1/fs)
    
    # Ensure frequencies outside [10,30]Hz are suppressed
    # For example, 5Hz component should be very small
    idx_5Hz = np.argmin(np.abs(freqs - 5))
    idx_40Hz = np.argmin(np.abs(freqs - 40))
    idx_20Hz = np.argmin(np.abs(freqs - 20))

    # The peak at 5Hz and 40Hz should be significantly smaller than at 20Hz
    # This is a qualitative check and might need tuning
    # assert fft_filtered[idx_5Hz] < fft_filtered[idx_20Hz] / 5 # Arbitrary ratio
    # assert fft_filtered[idx_40Hz] < fft_filtered[idx_20Hz] / 5 # Arbitrary ratio
    
    # Simpler assertion: the max magnitude should be around the band of interest
    band_indices = (freqs >= 10) & (freqs <= 30)
    out_of_band_indices = (freqs < 10) | (freqs > 30)
    
    # Assert that maximum outside band is much smaller than maximum inside band
    if np.any(band_indices):
        max_in_band = np.max(fft_filtered[band_indices])
        if np.any(out_of_band_indices):
            max_out_of_band = np.max(fft_filtered[out_of_band_indices])
            assert max_out_of_band < max_in_band * 0.1 # Example threshold
        else:
            # If no band indices, max_in_band could be 0, handle edge case
            assert np.allclose(filtered_signal, 0, atol=1e-5) # Should be all zeros if no band content
    
    def test_fast_kurtogram():
        fs = 1000
        t = np.linspace(0, 1, fs)
        # Signal with a transient (impulsive) component at 200 Hz
        signal = np.sin(2 * np.pi * 50 * t) 
        # Add a localized impulse
        signal[450:550] += 5 * np.random.randn(100)
        
        nlevel = 3
        Kwav, Level_w, freq_w, fc, bandwidth, max_kurt, level_max, c = fast_kurtogram(signal, fs, nlevel=nlevel)
        
        # Check output types
        assert isinstance(Kwav, np.ndarray)
        assert isinstance(Level_w, np.ndarray)
        assert isinstance(freq_w, np.ndarray)
        assert isinstance(fc, (float, np.float64, np.float32))
        assert isinstance(bandwidth, (float, np.float64, np.float32))
        assert isinstance(max_kurt, (float, np.float64, np.float32))
        assert isinstance(level_max, (float, np.float64, np.float32))
        assert isinstance(c, np.ndarray)
        
        # Check dimensions (Kwav shape: (2 * nlevel, 3 * 2**nlevel))
        assert Kwav.shape == (2 * nlevel, 3 * 2**nlevel)
        assert len(Level_w) == 2 * nlevel
        assert len(freq_w) == 3 * 2**nlevel
        
        # Check max_kurt is found
        assert max_kurt >= 0
        assert np.isclose(max_kurt, np.max(Kwav))
        
        # Check frequency range
        assert 0 < fc < fs / 2
        assert bandwidth > 0
    
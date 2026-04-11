import numpy as np
import scipy.stats as stats
from scipy.signal import welch
import pywt
from typing import Dict, Union, Any

class CrossDomainFeatureExtractor:
    """
    A robust feature extractor for rolling bearing vibration signals.
    Extracts 15 features across time, frequency, and time-frequency domains.
    Designed for cross-domain generalization (e.g., XJTU to Lab dataset) 
    by enforcing strict frequency bandwidth limitations.
    """

    def __init__(self, sampling_rate: float, max_freq_hz: float = 1280.0):
        """
        Initializes the feature extractor.

        Args:
            sampling_rate (float): The sampling frequency of the signal in Hz.
            max_freq_hz (float): The maximum frequency limit (bandwidth) to evaluate 
                                 during FFT. Default is 1280.0 Hz to match the 
                                 target domain's sensor limitations.
        """
        self.sampling_rate = sampling_rate
        self.max_freq_hz = max_freq_hz

    def extract_time_domain(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extracts 11 statistical features in the time domain.

        Args:
            signal (np.ndarray): 1D array of the raw vibration signal.

        Returns:
            Dict[str, float]: Dictionary containing time-domain features.
        """
        n = len(signal)
        mean_val = np.mean(signal)
        abs_signal = np.abs(signal)
        
        # 1. Root Mean Square
        rms = np.sqrt(np.mean(signal**2))
        
        # 2. Variance
        variance = np.var(signal, ddof=1)
        
        # 3. Peak Value
        peak_val = np.max(abs_signal)
        
        # 4. Kurtosis
        kurtosis = stats.kurtosis(signal, fisher=False)
        
        # 5. Crest Factor
        crest_factor = peak_val / rms if rms > 0 else 0
        
        # 6. Clearance Factor
        mean_sqrt_abs = np.mean(np.sqrt(abs_signal))
        clearance_factor = peak_val / (mean_sqrt_abs**2) if mean_sqrt_abs > 0 else 0
        
        # 7. Shape Factor
        mean_abs = np.mean(abs_signal)
        shape_factor = rms / mean_abs if mean_abs > 0 else 0
        
        # 8. Peak to Peak Value
        p2p_val = np.max(signal) - np.min(signal)
        
        # 9. Skewness
        skewness = stats.skew(signal)
        
        # 10. Impulse Factor
        impulse_factor = peak_val / mean_abs if mean_abs > 0 else 0
        
        # 11. Waveform Factor (Mathematically equivalent to Shape Factor in literature)
        waveform_factor = shape_factor 
        
        return {
            "td_rms": float(rms),
            "td_variance": float(variance),
            "td_peak_value": float(peak_val),
            "td_kurtosis": float(kurtosis),
            "td_crest_factor": float(crest_factor),
            "td_clearance_factor": float(clearance_factor),
            "td_shape_factor": float(shape_factor),
            "td_p2p_value": float(p2p_val),
            "td_skewness": float(skewness),
            "td_impulse_factor": float(impulse_factor),
            "td_waveform_factor": float(waveform_factor)
        }

    def extract_freq_domain(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extracts 3 frequency-domain features using FFT, STRICTLY applying 
        the maximum frequency limit (bandwidth) to prevent cross-domain mismatch.

        Args:
            signal (np.ndarray): 1D array of the raw vibration signal.

        Returns:
            Dict[str, float]: Dictionary containing frequency-domain features.
        """
        n = len(signal)
        
        # Perform Fast Fourier Transform (Real part only for efficiency)
        fft_values = np.fft.rfft(signal)
        fft_magnitudes = np.abs(fft_values) / n
        frequencies = np.fft.rfftfreq(n, d=1/self.sampling_rate)
        
        # --- BANDWIDTH FILTERING LIMITATION ---
        # Exclude all frequencies above the target sensor's max capability (1280 Hz)
        valid_idx = frequencies <= self.max_freq_hz
        filtered_magnitudes = fft_magnitudes[valid_idx]
        
        # 12. Peak to peak value of FFT
        if len(filtered_magnitudes) > 0:
            fft_p2p = np.max(filtered_magnitudes) - np.min(filtered_magnitudes)
        else:
            fft_p2p = 0.0
            
        # 13. Energy of FFT (Sum of squared magnitudes in the valid bandwidth)
        fft_energy = np.sum(filtered_magnitudes**2)
        
        # 14. Power Spectral Density (PSD)
        # Using Welch's method but limiting evaluation to the valid bandwidth
        freqs_welch, psd_welch = welch(signal, fs=self.sampling_rate, nperseg=256)
        valid_psd_idx = freqs_welch <= self.max_freq_hz
        filtered_psd = psd_welch[valid_psd_idx]
        psd_mean = np.mean(filtered_psd) if len(filtered_psd) > 0 else 0.0

        return {
            "fd_fft_p2p": float(fft_p2p),
            "fd_fft_energy": float(fft_energy),
            "fd_psd_mean": float(psd_mean)
        }

    def extract_time_freq_domain(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extracts 1 time-frequency domain feature using Continuous Wavelet Transform (CWT).
        Uses the Morlet wavelet.

        Args:
            signal (np.ndarray): 1D array of the raw vibration signal.

        Returns:
            Dict[str, float]: Dictionary containing the time-frequency feature.
        """
        # Define scales for CWT. Limit scales to prevent excessive computation.
        scales = np.arange(1, 31)
        
        # 15. Wavelet (CWT) using Morlet ('morl')
        # Returns coefficients matrix (scales x time)
        coefficients, _ = pywt.cwt(signal, scales, 'morl')
        
        # To represent the 2D CWT matrix as a 1D tabular feature, 
        # we calculate the total energy of the wavelet coefficients.
        cwt_energy = np.sum(np.abs(coefficients)**2)
        
        return {
            "tfd_cwt_energy": float(cwt_energy)
        }

    def extract_all_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Executes all feature extraction pipelines and combines them into a single record.

        Args:
            signal (np.ndarray): 1D array of the raw vibration signal.

        Returns:
            Dict[str, float]: A flattened dictionary containing all 15 features.
        """
        features = {}
        features.update(self.extract_time_domain(signal))
        features.update(self.extract_freq_domain(signal))
        features.update(self.extract_time_freq_domain(signal))
        
        return features

# ==========================================
# USAGE EXAMPLE FOR CROSS-DOMAIN PIPELINE
# ==========================================
if __name__ == "__main__":
    # Simulate a 1-second vibration signal from XJTU (25.6 kHz sampling rate)
    xjtu_sampling_rate = 25600.0
    simulated_xjtu_signal = np.random.randn(int(xjtu_sampling_rate))
    
    # Initialize extractor with constraint: max frequency 1280 Hz (Dosen's dataset limit)
    extractor = CrossDomainFeatureExtractor(
        sampling_rate=xjtu_sampling_rate, 
        max_freq_hz=1280.0
    )
    
    # Extract features
    extracted_features = extractor.extract_all_features(simulated_xjtu_signal)
    
    print("Cross-Domain Feature Extraction Successful:")
    for key, value in extracted_features.items():
        print(f"{key:>25}: {value:.6f}")
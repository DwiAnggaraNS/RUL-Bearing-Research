import numpy as np
import scipy.stats as stats
from typing import Dict

class CrossDomainFeatureExtractor:
    """
    A robust feature extractor for rolling bearing vibration signals.
    Extracts exactly 14 features (11 Time-Domain, 3 Frequency-Domain).
    Designed for cross-domain generalization (e.g., XJTU to Lab dataset) 
    by enforcing strict frequency bandwidth limitations and purging infinities.
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
        Extracts exactly 11 statistical features in the time domain.

        Args:
            signal (np.ndarray): 1D array of the raw vibration signal.

        Returns:
            Dict[str, float]: Dictionary containing time-domain features.
        """
        abs_signal = np.abs(signal)
        n = len(signal)
        
        # 1. Mean
        mean_val = np.mean(signal)
        
        # 2. Standard Deviation
        std_val = np.std(signal, ddof=1)
        
        # 3. RMS
        rms = np.sqrt(np.mean(signal**2))
        
        # 4. Peak Value
        peak_val = np.max(abs_signal)
        
        # 5. Skewness
        skewness = float(stats.skew(signal))
        
        # 6. Kurtosis
        kurtosis = float(stats.kurtosis(signal, fisher=False))
        
        # 7. Peak Factor (Peak Value / RMS)
        peak_factor = peak_val / rms if rms != 0 else 0.0
        
        # 8. Crest Factor (same as Peak Factor in many contexts, but requested separately, we'll implement standard CF)
        crest_factor = peak_val / rms if rms != 0 else 0.0
        
        # 9. Clearance Factor
        mean_sqrt_abs = np.mean(np.sqrt(abs_signal))
        clearance_factor = peak_val / (mean_sqrt_abs**2) if mean_sqrt_abs != 0 else 0.0
        
        # 10. Shape Factor
        mean_abs = np.mean(abs_signal)
        shape_factor = rms / mean_abs if mean_abs != 0 else 0.0
        
        # 11. Impulse Factor
        impulse_factor = peak_val / mean_abs if mean_abs != 0 else 0.0
        
        return {
            "td_mean": float(mean_val),
            "td_std": float(std_val),
            "td_rms": float(rms),
            "td_peak_value": float(peak_val),
            "td_skewness": float(skewness),
            "td_kurtosis": float(kurtosis),
            "td_peak_factor": float(peak_factor),
            "td_crest_factor": float(crest_factor),
            "td_clearance_factor": float(clearance_factor),
            "td_shape_factor": float(shape_factor),
            "td_impulse_factor": float(impulse_factor)
        }

    def extract_freq_domain(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extracts exactly 3 frequency-domain features using FFT, STRICTLY applying 
        the maximum frequency limit (bandwidth) to prevent cross-domain mismatch.

        Args:
            signal (np.ndarray): 1D array of the raw vibration signal.

        Returns:
            Dict[str, float]: Dictionary containing frequency-domain features.
        """
        n = len(signal)
        
        # Perform FFT (Real part only)
        fft_values = np.fft.rfft(signal)
        fft_magnitudes = np.abs(fft_values) / n
        frequencies = np.fft.rfftfreq(n, d=1/self.sampling_rate)
        
        # --- BANDWIDTH FILTERING LIMITATION ---
        # Exclude all frequencies above the target sensor's max capability (1280 Hz)
        valid_idx = frequencies <= self.max_freq_hz
        filtered_magnitudes = fft_magnitudes[valid_idx]
        filtered_freqs = frequencies[valid_idx]
        
        # Calculate features based on filtered bandwidth
        if len(filtered_magnitudes) > 0 and np.sum(filtered_magnitudes) > 0:
            # 12. Mean Frequency
            mean_frequency = np.sum(filtered_freqs * filtered_magnitudes) / np.sum(filtered_magnitudes)
            
            # 13. Centroid Frequency
            centroid_frequency = np.sum(filtered_freqs * (filtered_magnitudes ** 2)) / np.sum(filtered_magnitudes ** 2) if np.sum(filtered_magnitudes ** 2) > 0 else 0.0
            
            # 14. Peak to peak value of FFT
            fft_p2p = np.max(filtered_magnitudes) - np.min(filtered_magnitudes)
        else:
            mean_frequency = 0.0
            centroid_frequency = 0.0
            fft_p2p = 0.0

        return {
            "fd_mean_freq": float(mean_frequency),
            "fd_centroid_freq": float(centroid_frequency),
            "fd_fft_p2p": float(fft_p2p)
        }

    def extract_all_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Executes all feature extraction pipelines and combines them into a single record.
        Applies infinity purge to prevent 1.79e+308 explosions.

        Args:
            signal (np.ndarray): 1D array of the raw vibration signal.

        Returns:
            Dict[str, float]: A flattened dictionary containing exactly 14 features.
        """
        features = {}
        features.update(self.extract_time_domain(signal))
        features.update(self.extract_freq_domain(signal))
        
        # Apply nan_to_num to all features for Infinity Purge
        for k, v in features.items():
            features[k] = float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0))
            
        return features

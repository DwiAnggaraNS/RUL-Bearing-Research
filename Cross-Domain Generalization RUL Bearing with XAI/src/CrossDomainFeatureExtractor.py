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
        self.sampling_rate = sampling_rate
        self.max_freq_hz = max_freq_hz

    def extract_time_domain(self, signal: np.ndarray) -> Dict[str, float]:
        abs_signal = np.abs(signal)
        n = len(signal)
        
        mean_val = np.mean(signal) if n > 0 else 0.0
        std_val = np.std(signal, ddof=1) if n > 1 else 0.0
        rms = np.sqrt(np.mean(signal**2)) if n > 0 else 0.0
        peak_val = np.max(abs_signal) if n > 0 else 0.0
        p2p_val = np.max(signal) - np.min(signal) if n > 0 else 0.0
        skewness = float(stats.skew(signal)) if n > 1 else 0.0
        kurtosis = float(stats.kurtosis(signal, fisher=False)) if n > 1 else 0.0
        peak_factor = peak_val / rms if rms != 0 else 0.0
        mean_sqrt_abs = np.mean(np.sqrt(abs_signal)) if n > 0 else 0.0
        clearance_factor = peak_val / (mean_sqrt_abs**2) if mean_sqrt_abs != 0 else 0.0
        mean_abs = np.mean(abs_signal) if n > 0 else 0.0
        shape_factor = rms / mean_abs if mean_abs != 0 else 0.0
        impulse_factor = peak_val / mean_abs if mean_abs != 0 else 0.0
        
        return {
            "td_mean": float(mean_val),
            "td_std": float(std_val),
            "td_rms": float(rms),
            "td_peak_value": float(peak_val),
            "td_p2p": float(p2p_val),
            "td_skewness": float(skewness),
            "td_kurtosis": float(kurtosis),
            "td_peak_factor": float(peak_factor),
            "td_clearance_factor": float(clearance_factor),
            "td_shape_factor": float(shape_factor),
            "td_impulse_factor": float(impulse_factor)
        }

    def extract_freq_domain(self, signal: np.ndarray) -> Dict[str, float]:
        n = len(signal)
        if n == 0:
            return {"fd_mean_freq": 0.0, "fd_centroid_freq": 0.0, "fd_fft_p2p": 0.0}

        fft_values = np.fft.rfft(signal)
        fft_magnitudes = np.abs(fft_values) / n
        frequencies = np.fft.rfftfreq(n, d=1/self.sampling_rate)
        
        valid_idx = frequencies <= self.max_freq_hz
        filtered_magnitudes = fft_magnitudes[valid_idx]
        filtered_freqs = frequencies[valid_idx]
        
        if len(filtered_magnitudes) > 0 and np.sum(filtered_magnitudes) > 0:
            sum_mag = np.sum(filtered_magnitudes)
            mean_frequency = np.sum(filtered_freqs * filtered_magnitudes) / sum_mag
            sum_mag_sq = np.sum(filtered_magnitudes ** 2)
            centroid_frequency = np.sum(filtered_freqs * (filtered_magnitudes ** 2)) / sum_mag_sq if sum_mag_sq > 0 else 0.0
            fft_p2p = np.max(filtered_magnitudes) - np.min(filtered_magnitudes)
        else:
            mean_frequency, centroid_frequency, fft_p2p = 0.0, 0.0, 0.0

        return {
            "fd_mean_freq": float(mean_frequency),
            "fd_centroid_freq": float(centroid_frequency),
            "fd_fft_p2p": float(fft_p2p)
        }

    def extract_all_features(self, signal: np.ndarray) -> Dict[str, float]:
        features = {}
        features.update(self.extract_time_domain(signal))
        features.update(self.extract_freq_domain(signal))
        for k, v in features.items():
            features[k] = float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0))
        return features
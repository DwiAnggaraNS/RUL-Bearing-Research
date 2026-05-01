"""
CrossDomainFeatureExtractor.py
==============================
Physics-based rolling bearing vibration feature extractor.

Extracts exactly 14 features per 1024-sample signal window:
    - 11 time-domain statistics
    - 3 frequency-domain statistics (full 25.6 kHz spectrum; no frequency cap)

Design Notes:
    - The MAX_FREQ_HZ frequency cap has been permanently removed.
      The full 25.6 kHz spectrum is preserved to capture early-stage,
      high-frequency degradation signals (ball-pass harmonics, cage defects)
      that appear before low-frequency indicators begin to change.
    - All outputs are sanitised with np.nan_to_num to guarantee finite values
      before downstream aggregation.

Author: DwiAnggaraNS / PHM Research Team
PEP8: compliant
Emojis: none
"""

import numpy as np
import scipy.stats as stats
from typing import Dict


class CrossDomainFeatureExtractor:
    """
    Robust feature extractor for rolling bearing vibration signals.

    Extracts exactly 14 features:
        Time-domain  (11): mean, std, rms, peak_value, p2p, skewness,
                           kurtosis, peak_factor, clearance_factor,
                           shape_factor, impulse_factor.
        Frequency-domain (3): mean_frequency, centroid_frequency, fft_p2p.

    The full signal spectrum is used for frequency-domain features.
    No artificial frequency cap is applied.
    """

    def __init__(self, sampling_rate: float):
        """
        Initialise the extractor.

        Args:
            sampling_rate: ADC sampling rate in Hz (e.g. 25600.0 for XJTU-SY).
        """
        self.sampling_rate = sampling_rate

    def extract_time_domain(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Compute 11 time-domain statistical features from a raw signal window.

        Args:
            signal: 1-D array of raw vibration samples.

        Returns:
            Dictionary mapping feature name to scalar float value.
        """
        abs_signal = np.abs(signal)
        n = len(signal)

        mean_val         = np.mean(signal) if n > 0 else 0.0
        std_val          = np.std(signal, ddof=1) if n > 1 else 0.0
        rms              = np.sqrt(np.mean(signal ** 2)) if n > 0 else 0.0
        peak_val         = np.max(abs_signal) if n > 0 else 0.0
        p2p_val          = np.max(signal) - np.min(signal) if n > 0 else 0.0
        skewness         = float(stats.skew(signal)) if n > 1 else 0.0
        kurtosis         = float(stats.kurtosis(signal, fisher=False)) if n > 1 else 0.0
        peak_factor      = peak_val / rms if rms != 0.0 else 0.0
        mean_sqrt_abs    = np.mean(np.sqrt(abs_signal)) if n > 0 else 0.0
        clearance_factor = peak_val / (mean_sqrt_abs ** 2) if mean_sqrt_abs != 0.0 else 0.0
        mean_abs         = np.mean(abs_signal) if n > 0 else 0.0
        shape_factor     = rms / mean_abs if mean_abs != 0.0 else 0.0
        impulse_factor   = peak_val / mean_abs if mean_abs != 0.0 else 0.0

        return {
            "td_mean":             float(mean_val),
            "td_std":              float(std_val),
            "td_rms":              float(rms),
            "td_peak_value":       float(peak_val),
            "td_p2p":              float(p2p_val),
            "td_skewness":         float(skewness),
            "td_kurtosis":         float(kurtosis),
            "td_peak_factor":      float(peak_factor),
            "td_clearance_factor": float(clearance_factor),
            "td_shape_factor":     float(shape_factor),
            "td_impulse_factor":   float(impulse_factor),
        }

    def extract_freq_domain(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Compute 3 frequency-domain features using the full signal spectrum.

        The entire one-sided FFT magnitude spectrum is retained.  No
        artificial frequency cutoff is applied so that high-frequency
        degradation signatures are preserved.

        Args:
            signal: 1-D array of raw vibration samples.

        Returns:
            Dictionary with keys: fd_mean_freq, fd_centroid_freq, fd_fft_p2p.
        """
        n = len(signal)
        if n == 0:
            return {
                "fd_mean_freq":     0.0,
                "fd_centroid_freq": 0.0,
                "fd_fft_p2p":       0.0,
            }

        fft_values     = np.fft.rfft(signal)
        fft_magnitudes = np.abs(fft_values) / n
        frequencies    = np.fft.rfftfreq(n, d=1.0 / self.sampling_rate)

        sum_mag    = np.sum(fft_magnitudes)
        sum_mag_sq = np.sum(fft_magnitudes ** 2)

        mean_frequency     = float(np.sum(frequencies * fft_magnitudes) / sum_mag) if sum_mag > 0 else 0.0
        centroid_frequency = float(np.sum(frequencies * (fft_magnitudes ** 2)) / sum_mag_sq) if sum_mag_sq > 0 else 0.0
        fft_p2p            = float(np.max(fft_magnitudes) - np.min(fft_magnitudes))

        return {
            "fd_mean_freq":     mean_frequency,
            "fd_centroid_freq": centroid_frequency,
            "fd_fft_p2p":       fft_p2p,
        }

    def extract_all_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extract all 14 features and sanitise against NaN / Inf values.

        Args:
            signal: 1-D array of raw vibration samples.

        Returns:
            Dictionary of 14 finite float feature values.
        """
        features: Dict[str, float] = {}
        features.update(self.extract_time_domain(signal))
        features.update(self.extract_freq_domain(signal))
        features = {
            k: float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0))
            for k, v in features.items()
        }
        return features
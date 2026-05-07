import os
import glob
import numpy as np
import pandas as pd
import scipy.stats
from scipy.fft import fft, fftfreq
from typing import Dict, Tuple, List

class FeatureExtractionAndSelection:
    """
    A class to handle feature extraction from raw vibration data (acceleration or velocity),
    compute evaluation metrics, and perform lenient feature selection based on the 
    meta-probability of Spearman correlation and Modified Monotonicity.
    """

    def __init__(self, data_directory: str):
        """
        Initializes the feature extraction parameters based on bearing specifications.
        
        Args:
            data_directory (str): The base path to the directory containing conditions (e.g., Condition_1, Condition_2).
                                  Directory structure: data_directory / condition / bearing / *.csv
        """
        self.data_dir = data_directory
        
        # Bearing Dimensions for XJTU bearing dataset
        self.ball_diameter_mm = 7.92
        self.mean_bearing_diameter_mm = 34.55
        self.num_balls = 8
        self.sampling_frequency_hz = 25.6e3
        
        # Shaft frequencies in Hz for the 15 bearings (5 bearings per condition)
        # Condition 1: 35.0 Hz, Condition 2: 37.5 Hz, Condition 3: 40.0 Hz
        self.shaft_frequencies = {
            1: 35.0,
            2: 37.5,
            3: 40.0
        }

    def _extract_time_features(self, signal: np.ndarray) -> np.ndarray:
        """
        Extracts 7 time-domain features from a given 1D vibration signal.
        The features are: Max Amplitude, RMS, Kurtosis, Shape Factor, Skewness, Impulse Factor, Crest Factor.
        """
        features = np.zeros(7)
        abs_sig = np.abs(signal)
        
        features[0] = np.max(abs_sig)
        features[1] = np.sqrt(np.mean(signal**2))  # RMS
        features[2] = scipy.stats.kurtosis(signal, fisher=False)
        
        mean_abs = np.mean(abs_sig)
        features[3] = features[1] / mean_abs if mean_abs != 0 else 0  # Shape factor
        features[4] = scipy.stats.skew(signal)
        features[5] = features[0] / mean_abs if mean_abs != 0 else 0  # Impulse factor
        features[6] = features[0] / features[1] if features[1] != 0 else 0  # Crest factor
        
        return features

    def _extract_frequency_features(self, signal: np.ndarray, shaft_freq: float) -> np.ndarray:
        """
        Extracts 24 frequency-domain features (16 fault/overall energy + 8 evenly divided one-sided spectrum bands).
        """
        m = len(signal)
        df = self.sampling_frequency_hz / m
        
        # Compute fault frequencies
        BPFO = self.num_balls * shaft_freq / 2 * (1 - self.ball_diameter_mm / self.mean_bearing_diameter_mm)
        BPFI = self.num_balls * shaft_freq / 2 * (1 + self.ball_diameter_mm / self.mean_bearing_diameter_mm)
        BSF = (self.mean_bearing_diameter_mm / (2 * self.ball_diameter_mm)) * \
              (1 - (self.ball_diameter_mm / self.mean_bearing_diameter_mm)**2) * shaft_freq
              
        # Perform FFT
        Y = fft(np.abs(signal))
        P2 = np.abs(Y / m)
        P1 = P2[:m//2 + 1]
        P1[1:-1] = 2 * P1[1:-1]
        
        features = np.zeros(24)
        
        def calculate_band_energy(center_freq: float, margin: float = 0.05) -> float:
            si = int((1 - margin) * center_freq / df)
            ei = int((1 + margin) * center_freq / df)
            si = max(0, min(si, len(P1)-1))
            ei = max(0, min(ei, len(P1)-1))
            return np.sqrt(np.sum(P1[si:ei]**2) / 2)

        # BPFO harmonics
        features[0] = calculate_band_energy(BPFO)
        features[1] = calculate_band_energy(2 * BPFO)
        features[2] = calculate_band_energy(3 * BPFO)
        features[3] = np.sqrt(features[0]**2 + features[1]**2 + features[2]**2)
        
        # BPFI harmonics
        features[4] = calculate_band_energy(BPFI)
        features[5] = calculate_band_energy(2 * BPFI)
        features[6] = calculate_band_energy(3 * BPFI)
        features[7] = np.sqrt(features[4]**2 + features[5]**2 + features[6]**2)
        
        # BSF harmonics
        features[8] = calculate_band_energy(BSF)
        features[9] = calculate_band_energy(2 * BSF)
        features[10] = calculate_band_energy(3 * BSF)
        features[11] = np.sqrt(features[8]**2 + features[9]**2 + features[10]**2)
        
        # Overall energies
        si_overall = int(shaft_freq * 0.8 / df)
        features[12] = np.sqrt(np.sum(P1[si_overall:]**2) / 2)
        
        si_bearing = int(shaft_freq * 2.1 / df)
        features[13] = np.sqrt(np.sum(P1[si_bearing:]**2) / 2)
        
        si_low = int(shaft_freq * 0.5 / df)
        ei_low = int(400 / df)
        features[14] = np.sqrt(np.sum(P1[si_low:ei_low]**2) / 2)
        
        si_low_bearing = int(shaft_freq * 2.1 / df)
        features[15] = np.sqrt(np.sum(P1[si_low_bearing:ei_low]**2) / 2)
        
        # 8 evenly divided portions of the one-sided FFT spectrum (not wavelets)
        P1[0] = 0 # Remove DC
        band_size = (self.sampling_frequency_hz / 2) / 8
        for n_band in range(1, 9):
            si_b = int((n_band - 1) * band_size / df)
            if si_b == 0: si_b = 1  # Avoid DC again
            ei_b = int(n_band * band_size / df)
            si_b = max(0, min(si_b, len(P1)-1))
            ei_b = max(0, min(ei_b, len(P1)-1))
            features[16 + (n_band - 1)] = np.sqrt(np.sum(P1[si_b:ei_b]**2) / 2)
            
        return features

    def process_csv_files(self) -> Dict[str, np.ndarray]:
        """
        Iterates over the hierarchical directory structure and extracts features 
        from all available CSV files.
        
        Returns:
            Dict[str, np.ndarray]: A dictionary mapping a string identifier (e.g., 'Condition_1_Bearing_1')
                                   to a 2D numpy array of shape (num_files, 124).
                                   Each row represents features extracted from one CSV file.
        """
        import scipy.signal
        from scipy.integrate import cumulative_trapezoid
        extracted_data = {}
        
        # Traverse conditions
        condition_folders = glob.glob(os.path.join(self.data_dir, "*"))
        for condition_folder in condition_folders:
            if not os.path.isdir(condition_folder):
                continue
                
            condition_name = os.path.basename(condition_folder)
            
            # Determine shaft frequency from condition name (assuming Condition_1, Condition_2, etc.)
            try:
                cond_idx = int(''.join(filter(str.isdigit, condition_name)))
                shaft_freq = self.shaft_frequencies.get(cond_idx, 35.0)
            except ValueError:
                shaft_freq = 35.0
                
            bearing_folders = glob.glob(os.path.join(condition_folder, "*"))
            for bearing_folder in bearing_folders:
                if not os.path.isdir(bearing_folder):
                    continue
                    
                bearing_name = os.path.basename(bearing_folder)
                identifier = f"{condition_name}_{bearing_name}"
                
                csv_files = sorted(glob.glob(os.path.join(bearing_folder, "*.csv")))
                
                bearing_features = []
                for csv_file in csv_files:
                    try:
                        df_data = pd.read_csv(csv_file, header=None)
                        if df_data.shape[1] >= 2:
                            # Assuming horizontal acceleration at col 0, vertical acceleration at col 1
                            ha_acc_signal = df_data.iloc[:, 0].values
                            va_acc_signal = df_data.iloc[:, 1].values
                            
                            # Derive velocity from acceleration numerically (prevent drift by detrending)
                            ha_vel_signal = scipy.signal.detrend(
                                cumulative_trapezoid(scipy.signal.detrend(ha_acc_signal), dx=1/self.sampling_frequency_hz, initial=0)
                            )
                            va_vel_signal = scipy.signal.detrend(
                                cumulative_trapezoid(scipy.signal.detrend(va_acc_signal), dx=1/self.sampling_frequency_hz, initial=0)
                            )
                            
                            # Extract horizontal direction features (Acc & Vel) = 31 + 31 = 62 features
                            ha_acc_time = self._extract_time_features(ha_acc_signal)
                            ha_acc_freq = self._extract_frequency_features(ha_acc_signal, shaft_freq)
                            ha_vel_time = self._extract_time_features(ha_vel_signal)
                            ha_vel_freq = self._extract_frequency_features(ha_vel_signal, shaft_freq)
                            
                            # Extract vertical direction features (Acc & Vel) = 31 + 31 = 62 features
                            va_acc_time = self._extract_time_features(va_acc_signal)
                            va_acc_freq = self._extract_frequency_features(va_acc_signal, shaft_freq)
                            va_vel_time = self._extract_time_features(va_vel_signal)
                            va_vel_freq = self._extract_frequency_features(va_vel_signal, shaft_freq)
                            
                            # Combine all 124 features per CSV structure 
                            file_features = np.concatenate([
                                ha_acc_time, ha_acc_freq, ha_vel_time, ha_vel_freq,
                                va_acc_time, va_acc_freq, va_vel_time, va_vel_freq
                            ])
                            bearing_features.append(file_features)
                            
                    except Exception as e:
                        print(f"Error processing {csv_file}: {e}")
                        
                if bearing_features:
                    extracted_data[identifier] = np.array(bearing_features)
                    
        return extracted_data

    @staticmethod
    def calculate_modified_monotonicity(feature_series: np.ndarray, sigma: float) -> float:
        """
        Calculates the modified monotonicity while accounting for measurement noise.
        
        Args:
            feature_series (np.ndarray): The 1D array representing the feature time series.
            sigma (float): The estimated noise level in the time series measurement.
            
        Returns:
            float: The modified monotonicity score.
        """
        feature_series = np.asarray(feature_series, dtype=float)
        
        # Determine if the feature is generally increasing or decreasing
        # Evaluates the averages of the first two and last two points
        end_avg = (feature_series[-1] + feature_series[-2]) / 2.0
        start_avg = (feature_series + feature_series[9]) / 2.0
        
        if end_avg >= start_avg:
            myf = feature_series
        else:
            # Invert to make it monotonically increasing
            myf = np.max(feature_series) - feature_series
            
        epsilon = 1e-2
        dF = np.diff(myf)
        
        # Compute alpha parameter
        my_alpha = np.arctanh(1 - epsilon) / (epsilon + sigma)
        
        # Calculate metric components
        allc = np.tanh((dF + sigma) * my_alpha)
        
        # Compute final monotonicity
        # Prevent division by zero
        sum_abs_allc = np.sum(np.abs(allc))
        if sum_abs_allc == 0:
            return 0.0
            
        modified_monotonicity = np.sum(allc) / sum_abs_allc
        return float(modified_monotonicity)

    @staticmethod
    def calculate_meta_probability(metric_values: np.ndarray, cutoff_range: Tuple[float, float]) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Calculates the meta-probability for a given set of metric scores across a population.
        
        Args:
            metric_values (np.ndarray): Array of metric scores (e.g., monotonicity scores) for all units.
            cutoff_range (Tuple[float, float]): The lower and upper cutoffs (e.g., (0.4, 1.0)).
            
        Returns:
            Tuple[float, np.ndarray, np.ndarray]: 
                - The calculated meta-probability.
                - The vector of cutoff values evaluated.
                - The metric curve used for meta-probability.
        """
        metric_values = np.asarray(metric_values, dtype=float)
        n_bearings = len(metric_values)
        
        # Define 100 evaluation points between lower and upper cutoffs
        cutoff_vec = np.linspace(cutoff_range, cutoff_range[9], 100)
        metric_vec = np.zeros(len(cutoff_vec))
        
        # Evaluate how many units exceed each cutoff
        for j in range(len(cutoff_vec)):
            metric_vec[j] = np.sum(metric_values > cutoff_vec[j])
            
        # Calculate survival probability integral using trapezoidal rule
        integral_val = np.trapz(metric_vec, x=cutoff_vec)
        meta_probability = integral_val / (cutoff_range[9] - cutoff_range) / n_bearings
        
        return float(meta_probability), cutoff_vec, metric_vec

    @staticmethod
    def perform_lenient_feature_selection(spearman_meta_probs: np.ndarray, monotonicity_meta_probs: np.ndarray) -> np.ndarray:
        """
        Executes the lenient feature selection to remove noisy features. Retains features 
        where both Spearman and Modified Monotonicity meta-probabilities exceed the 40th percentile.
        
        Args:
            spearman_meta_probs (np.ndarray): Meta-probability scores of the Spearman correlation for all features.
            monotonicity_meta_probs (np.ndarray): Meta-probability scores of the Modified Monotonicity for all features.
            
        Returns:
            np.ndarray: An array containing the indices of the selected features that passed the criteria.
        """
        spearman_meta_probs = np.asarray(spearman_meta_probs)
        monotonicity_meta_probs = np.asarray(monotonicity_meta_probs)
        
        # Calculate the 40% quantile (40th percentile) thresholds
        spearman_threshold = np.percentile(spearman_meta_probs, 40)
        monotonicity_threshold = np.percentile(monotonicity_meta_probs, 40)
        
        # Select indices where both metrics are strictly strictly greater than the threshold
        selected_feature_indices = np.where(
            (spearman_meta_probs > spearman_threshold) & 
            (monotonicity_meta_probs > monotonicity_threshold)
        )
        
        return selected_feature_indices
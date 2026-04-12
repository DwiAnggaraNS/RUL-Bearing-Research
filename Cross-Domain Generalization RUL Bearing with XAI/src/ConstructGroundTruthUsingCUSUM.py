import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional

class UnivariateCUSUMDetector:
    """
    Cumulative Sum (CUSUM) Detector for univariate time-series data.
    Designed to track gradual degradation and detect early fault change points 
    in bearing datasets (e.g., RMS or Kurtosis features).
    """

    def __init__(self, 
                 baseline_ratio: float = 0.1, 
                 k_factor: float = 0.5, 
                 h_factor: float = 5.0):
        """
        Initializes the CUSUM detector.

        Args:
            baseline_ratio (float): The proportion of initial data assumed to be 
                                    in a healthy state (used to calculate mu_0 and sigma_0).
            k_factor (float): The drift threshold multiplier (slack value).
            h_factor (float): The decision threshold multiplier to trigger an alarm.
        """
        self.baseline_ratio = baseline_ratio
        self.k_factor = k_factor
        self.h_factor = h_factor
        
        # Internal state variables to store results for plotting
        self.feature_data: Optional[np.ndarray] = None
        self.cusum_scores: Optional[np.ndarray] = None
        self.change_point: Optional[int] = None
        self.h_threshold: Optional[float] = None

    def fit_predict(self, feature_series: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Executes the CUSUM algorithm to detect the degradation change point.

        Args:
            feature_series (np.ndarray): 1D array of the extracted feature (1 row = 1 min).
                                         e.g., Array of RMS or Kurtosis values.

        Returns:
            Tuple[int, np.ndarray]: 
                - tcp_index: The index (minute) where degradation officially starts.
                - cusum_scores: The computed CUSUM trajectory array.
        """
        self.feature_data = np.array(feature_series)
        n_samples = len(self.feature_data)
        
        # 1. Define the healthy baseline
        baseline_len = max(1, int(n_samples * self.baseline_ratio))
        baseline_data = self.feature_data[:baseline_len]
        
        mu_0 = np.mean(baseline_data)
        sigma_0 = np.std(baseline_data)
        
        # 2. Set CUSUM Parameters based on baseline statistics
        k = self.k_factor * sigma_0
        self.h_threshold = self.h_factor * sigma_0
        
        # 3. Initialize CUSUM tracking arrays
        self.cusum_scores = np.zeros(n_samples)
        self.change_point = n_samples - 1  # Default to end if no fault is detected
        fault_detected = False
        
        # 4. Upper CUSUM Calculation Loop
        for t in range(1, n_samples):
            # Accumulate deviation from healthy mean
            deviation = self.feature_data[t] - mu_0 - k
            self.cusum_scores[t] = max(0, self.cusum_scores[t-1] + deviation)
            
            # Check if accumulated score exceeds the decision threshold
            if self.cusum_scores[t] > self.h_threshold and not fault_detected:
                self.change_point = t
                fault_detected = True
                
        print(f"Algorithm Execution Finished.")
        if fault_detected:
            print(f"SUCCESS: Change Point (Degradation) detected at Minute: {self.change_point}")
        else:
            print("WARNING: No degradation detected. The feature remained within normal bounds.")
            
        return self.change_point, self.cusum_scores

    def plot_degradation(self, feature_name: str = "Extracted Feature (e.g., RMS/Kurtosis)"):
        """
        Generates a visualization of the feature trajectory.
        Highlights the Change Point and colors the degradation state in red.

        Args:
            feature_name (str): The label for the Y-axis.
        """
        if self.feature_data is None or self.change_point is None:
            raise ValueError("Error: You must run fit_predict() before plotting.")

        time_steps = np.arange(len(self.feature_data))
        
        plt.figure(figsize=(14, 6))
        
        # Plot Healthy State (Blue)
        plt.plot(time_steps[:self.change_point + 1], 
                 self.feature_data[:self.change_point + 1], 
                 color='blue', linewidth=2, label='Healthy State')
        
        # Plot Degradation State (Red)
        plt.plot(time_steps[self.change_point:], 
                 self.feature_data[self.change_point:], 
                 color='red', linewidth=2, label='Degradation State')
        
        # Highlight the Change Point
        plt.axvline(x=self.change_point, color='black', linestyle='--', linewidth=2, 
                    label=f'Change Point ($T_e$) = {self.change_point}')
        
        # Plot formatting
        plt.title(f'Machine Degradation Tracking using CUSUM\nFeature: {feature_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Time (Minutes)', fontsize=12)
        plt.ylabel(feature_name, fontsize=12)
        plt.legend(loc='upper left', fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

# ==========================================
# USAGE EXAMPLE FOR YOUR PIPELINE
# ==========================================
if __name__ == "__main__":
    # Simulate a feature array (e.g., 500 minutes of RMS or Kurtosis data)
    # Healthy for the first 300 minutes, then gradually degrading
    np.random.seed(42)
    healthy_phase = np.random.normal(loc=0.5, scale=0.05, size=300)
    degradation_phase = healthy_phase[-200:] + np.linspace(0, 0.8, 200) # Gradual increase
    simulated_feature_1D = np.concatenate([healthy_phase, degradation_phase])
    
    # 1. Initialize Detector
    detector = UnivariateCUSUMDetector(baseline_ratio=0.1, k_factor=0.5, h_factor=5.0)
    
    # 2. Run Detection
    tcp, cusum_trajectory = detector.fit_predict(simulated_feature_1D)
    
    # 3. Visualize
    detector.plot_degradation(feature_name="RMS Value")
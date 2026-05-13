import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class UnivariateCUSUMDetector:
    """
    Cumulative Sum (CUSUM) Detector for univariate time-series data.
    Designed to track gradual degradation and detect early fault change points
    in bearing datasets (e.g., RMS or Kurtosis features).

    Refactored with conservative parameters and robustness guards to prevent
    premature triggering on short bearings or low-variance baselines.

    Changes from original:
        - baseline_ratio default raised to 0.2 (was 0.1)
        - h_factor default raised to 8.0 (was 5.0) for conservative detection
        - baseline_len floored at 20 samples minimum
        - sigma_0 fallback to global std when baseline variance is near-zero
    """

    def __init__(
        self,
        baseline_ratio: float = 0.2,
        k_factor: float = 0.5,
        h_factor: float = 8.0
    ):
        """
        Initializes the CUSUM detector with conservative defaults.

        Args:
            baseline_ratio: Proportion of initial data assumed healthy
                            (used to estimate mu_0 and sigma_0).
            k_factor: Drift threshold multiplier (slack value).
            h_factor: Decision threshold multiplier to trigger an alarm.
                      Higher values require more accumulated evidence,
                      reducing false early detections.
        """
        self.baseline_ratio = baseline_ratio
        self.k_factor = k_factor
        self.h_factor = h_factor

        # Internal state variables for results and plotting
        self.feature_data: Optional[np.ndarray] = None
        self.cusum_scores: Optional[np.ndarray] = None
        self.change_point: Optional[int] = None
        self.h_threshold: Optional[float] = None

    def fit_predict(self, feature_series: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Executes the CUSUM algorithm to detect the degradation change point.

        Includes robustness guards:
            - Baseline length is floored at 20 samples to ensure stable
              estimation of mu_0 and sigma_0 for short bearings.
            - If sigma_0 from the baseline is near-zero (< 1e-6), falls back
              to the global standard deviation of the entire series.

        Args:
            feature_series: 1D array of the extracted feature values.
                            Ideally pre-smoothed (e.g., rolling mean of RMS).

        Returns:
            Tuple of:
                - tcp_index: Index where degradation is detected.
                - cusum_scores: The computed CUSUM trajectory array.
        """
        self.feature_data = np.array(feature_series, dtype=np.float64)
        n_samples = len(self.feature_data)

        # 1. Define the healthy baseline with floor guard
        baseline_len = max(20, int(n_samples * self.baseline_ratio))
        baseline_len = min(baseline_len, n_samples)
        baseline_data = self.feature_data[:baseline_len]

        mu_0 = np.mean(baseline_data)
        sigma_0 = np.std(baseline_data)

        # Guard: if baseline variance is near-zero, use global std
        if sigma_0 < 1e-6:
            sigma_0 = np.std(self.feature_data)
            if sigma_0 < 1e-6:
                # Entire series is constant; no degradation detectable
                print("WARNING: Feature series has near-zero variance. "
                      "No meaningful change point can be detected.")
                self.cusum_scores = np.zeros(n_samples)
                self.change_point = n_samples - 1
                return self.change_point, self.cusum_scores

        # 2. Set CUSUM parameters from baseline statistics
        k = self.k_factor * sigma_0
        self.h_threshold = self.h_factor * sigma_0

        # 3. Initialize CUSUM tracking
        self.cusum_scores = np.zeros(n_samples)
        self.change_point = n_samples - 1
        fault_detected = False

        # 4. Upper CUSUM calculation loop
        for t in range(1, n_samples):
            deviation = self.feature_data[t] - mu_0 - k
            self.cusum_scores[t] = max(0, self.cusum_scores[t - 1] + deviation)

            if self.cusum_scores[t] > self.h_threshold and not fault_detected:
                self.change_point = t
                fault_detected = True

        print("CUSUM execution finished.")
        if fault_detected:
            print(f"  Change point detected at index: {self.change_point}")
        else:
            print("  No degradation detected within threshold bounds.")

        return self.change_point, self.cusum_scores

    def plot_degradation(
        self, feature_name: str = "Extracted Feature (e.g., RMS/Kurtosis)"
    ):
        """
        Generates a visualization of the feature trajectory.
        Highlights the Change Point and colors the degradation phase in red.

        Args:
            feature_name: Label for the Y-axis.
        """
        if self.feature_data is None or self.change_point is None:
            raise ValueError(
                "Error: You must run fit_predict() before plotting."
            )

        time_steps = np.arange(len(self.feature_data))

        plt.figure(figsize=(14, 6))

        # Plot Healthy State (Blue)
        plt.plot(
            time_steps[:self.change_point + 1],
            self.feature_data[:self.change_point + 1],
            color='blue', linewidth=2, label='Healthy State'
        )

        # Plot Degradation State (Red)
        plt.plot(
            time_steps[self.change_point:],
            self.feature_data[self.change_point:],
            color='red', linewidth=2, label='Degradation State'
        )

        # Highlight the Change Point
        plt.axvline(
            x=self.change_point, color='black', linestyle='--', linewidth=2,
            label=f'Change Point ($T_e$) = {self.change_point}'
        )

        plt.title(
            f'Machine Degradation Tracking using CUSUM\n'
            f'Feature: {feature_name}',
            fontsize=14, fontweight='bold'
        )
        plt.xlabel('Time (Minutes)', fontsize=12)
        plt.ylabel(feature_name, fontsize=12)
        plt.legend(loc='upper left', fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
    
    def detect_change_point_sigma(self, signal, window_size=10, threshold=3.0):
        """
        Deteksi Change Point menggunakan Sliding Window 3-Sigma.
        Metode ini lebih stabil terhadap noise sesaat (spike) dibandingkan CUSUM murni.
        
        Args:
            signal (np.ndarray): Array 1D data fitur (RMS/Composite).
            window_size (int): Jumlah sampel sebelumnya untuk menghitung statistik lokal.
            threshold (float): Perkalian sigma (default 3.0).
            
        Returns:
            int: Index di mana degradasi dimulai.
        """
        signal = np.asarray(signal)
        n = len(signal)
        self.change_point = n - 1

        # Guard: Jika data terlalu pendek, kembalikan indeks terakhir
        if n < window_size:
            return n - 1
            
        # Mulai scanning setelah window_size terlewati
        for i in range(window_size, n):
            # 1. Ambil data historis (window) untuk baseline lokal
            history = signal[i - window_size : i]
            
            mu = np.mean(history)
            sigma = np.std(history)
            
            # Guard: Hindari pembagian nol jika data flat sempurna
            if sigma < 1e-9:
                sigma = 1e-9
                
            # 2. Hitung batas atas (Upper Control Limit)
            limit = mu + (threshold * sigma)
            
            # 3. Cek apakah titik saat ini melampaui batas
            if signal[i] > limit:
                # 4. Validasi Tambahan: Pastikan ini bukan noise sesaat (spike)
                # Cek apakah rata-rata 3 poin ke depan juga tetap tinggi
                if i + 3 < n:
                    if np.mean(signal[i : i+3]) > mu:
                        self.change_point = i
                        print(f"  Change point detected at index: {self.change_point}")    
                        return i
                else:
                    # Jika mendekati akhir data, anggap valid jika tembus batas
                    return i
        # Jika tidak terdeteksi sama sekali, kembalikan indeks terakhir
        return n - 1

    def generate_pronostia_piecewise_rul(self, feature_series: np.ndarray, ema_span: int = 10) -> Tuple[np.ndarray, int]:
        """
        Generate Piecewise Linear RUL target specifically for PRONOSTIA dataset context.
        Applies Exponential Moving Average (EMA) to smooth the signal, detects Change Point (TCP),
        and creates an RUL array that is 1.0 (Healthy) until TCP, then linearly degrades to 0.0.

        Args:
            feature_series (np.ndarray): 1D array of extracted feature (e.g., RMS or Kurtosis).
            ema_span (int): Span for the Pandas Exponential Moving Average.

        Returns:
            Tuple[np.ndarray, int]: 
                - Piecewise RUL array (same length as input).
                - TCP index (Time to Change Point).
        """
        feature_series = np.asarray(feature_series, dtype=np.float64)
        n = len(feature_series)
        
        # 1. Data Smoothing (EMA)
        smoothed_series = pd.Series(feature_series).ewm(span=ema_span).mean().values
        
        # 2. CUSUM Detection
        tcp_index, _ = self.fit_predict(smoothed_series)
        
        # Fallback safety: If TCP is detected near the end or fails to detect effectively
        if tcp_index >= n - 1 or tcp_index < max(20, int(n * 0.05)):
            # Set to 10% of total length as a conservative fallback
            tcp_index = max(1, int(n * 0.10))
            
        # 3. Construct Piecewise Linear RUL
        # Flat 1.0 until TCP, then linear degradation to 0.0
        rul_target = np.ones(n, dtype=np.float64)
        if tcp_index < n - 1:
            degradation_length = n - tcp_index
            rul_target[tcp_index:] = np.linspace(1.0, 0.0, degradation_length)
            
        return rul_target, int(tcp_index), smoothed_series

    def plot_cusum_validation(self, feature_series: np.ndarray, smoothed_series: np.ndarray, rul_target: np.ndarray, tcp_index: int):
        """
        Plots validation dual-axis visualization of original/smoothed feature against Constructed Piecewise RUL.
        """
        time_steps = np.arange(len(feature_series))

        fig, ax1 = plt.subplots(figsize=(12, 5))
        
        color1 = 'tab:gray'
        color2 = 'tab:blue'
        
        ax1.set_xlabel('Time (Minutes or Windows)')
        ax1.set_ylabel('Vibration Feature Level', color=color2)
        ax1.plot(time_steps, feature_series, color=color1, alpha=0.4, label='Raw Feature')
        ax1.plot(time_steps, smoothed_series, color=color2, linewidth=2, label=f'EMA Smoothed Feature')
        
        # Highlight TCP
        ax1.axvline(x=tcp_index, color='red', linestyle='--', linewidth=2, label=f'TCP / FPT (Index {tcp_index})')
        ax1.tick_params(axis='y', labelcolor=color2)
        ax1.legend(loc='upper left')
        
        ax2 = ax1.twinx()
        color3 = 'tab:green'
        ax2.set_ylabel('RUL Target (Degradation State)', color=color3)
        ax2.plot(time_steps, rul_target, color=color3, linestyle='-.', linewidth=3, label='Piecewise RUL Target')
        ax2.tick_params(axis='y', labelcolor=color3)
        ax2.legend(loc='upper right')

        plt.title('Piecewise Linear RUL Construction via CUSUM Change Point Detection')
        fig.tight_layout()
        plt.show()

# ==========================================
# USAGE EXAMPLE FOR YOUR PIPELINE
# ==========================================
if __name__ == "__main__":
    # Simulate a feature array (500 minutes of RMS data)
    # Healthy for 300 minutes, then gradually degrading
    np.random.seed(42)
    healthy_phase = np.random.normal(loc=0.5, scale=0.05, size=300)
    degradation_phase = healthy_phase[-200:] + np.linspace(0, 0.8, 200)
    simulated_feature_1D = np.concatenate([healthy_phase, degradation_phase])

    # 1. Initialize Detector (conservative defaults)
    detector = UnivariateCUSUMDetector(
        baseline_ratio=0.2, k_factor=0.5, h_factor=8.0
    )

    # 2. Run Detection
    tcp, cusum_trajectory = detector.fit_predict(simulated_feature_1D)

    # 3. Visualize
    detector.plot_degradation(feature_name="RMS Value")
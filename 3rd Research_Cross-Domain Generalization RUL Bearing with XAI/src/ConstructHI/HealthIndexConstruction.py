import numpy as np
import scipy.stats as stats
from scipy.stats import qmc
from typing import List, Dict, Tuple, Callable

# Note: Assuming FeatureExtractionAndSelection is available in the module 
# to import the get_metaprobability method as requested.
# from feature_extraction import FeatureExtractionAndSelection

class HealthIndexConstruction:
    """
    A class dedicated to generating optimal weights for feature fusion 
    and constructing the final Health Index (HI) using a multi-objective 
    optimization scheme across a population of units.
    """

    def __init__(self, num_samples_lhs: int = 2000):
        """
        Initializes the HI Construction environment.
        
        Args:
            num_samples_lhs (int): Number of weight combinations to generate using 
                                   Latin Hypercube Sampling (default is 2000).
        """
        self.num_samples_lhs = num_samples_lhs

    @staticmethod
    def generate_lhs_weights(num_features: int, num_samples: int) -> np.ndarray:
        """
        Generates weight combinations using Latin Hypercube Sampling (LHS) bounded 
        between -1 and 1, ensuring the sum of absolute weights equals 1.
        
        Args:
            num_features (int): The number of features passing the lenient selection.
            num_samples (int): The number of LHS samples to generate.
            
        Returns:
            np.ndarray: A matrix of shape (num_samples, num_features) containing normalized weights.
        """
        # Generate LHS samples uniformly between [0, 1)
        sampler = qmc.LatinHypercube(d=num_features)
        sample = sampler.random(n=num_samples)
        
        # Scale samples to bounds [-1, 1]
        scaled_sample = qmc.scale(sample, [-1.0] * num_features, [1.0] * num_features)
        
        # Apply constraint: sum(|w|) <= 1. For simplicity, we normalize to exactly 1.
        norms = np.sum(np.abs(scaled_sample), axis=1, keepdims=True)
        normalized_weights = scaled_sample / np.where(norms == 0, 1e-10, norms)
        
        return normalized_weights

    @staticmethod
    def get_metrics(hi_signal: np.ndarray, fpt_index: int, 
                    mod_monotonicity_func: Callable) -> Dict[str, float]:
        """
        Calculates signal performance metrics for a single Health Index degradation trajectory.
        
        Args:
            hi_signal (np.ndarray): The constructed 1D Health Index signal.
            fpt_index (int): First Prediction Time index denoting the onset of degradation.
            mod_monotonicity_func (Callable): Function to compute modified monotonicity.
            
        Returns:
            Dict[str, float]: A dictionary containing Pearson, Spearman, Old Monotonicity, 
                              Modified Monotonicity, and Robustness values.
        """
        # Focus strictly on the degradation phase (from FPT to EOL)
        degradation_phase = hi_signal[fpt_index:]
        m = len(degradation_phase)
        
        if m < 2:
            return {"pearson": 0.0, "spearman": 0.0, "old_mon": 0.0, "mod_mon": 0.0, "robustness": 0.0}
            
        time_steps = np.arange(m)
        
        # 1. Pearson & Spearman Correlations
        pearson_corr, _ = stats.pearsonr(degradation_phase, time_steps)
        spearman_corr, _ = stats.spearmanr(degradation_phase, time_steps)
        
        # 2. Old Monotonicity
        diffs = np.diff(degradation_phase)
        pos_steps = np.sum(diffs >= 0)
        neg_steps = np.sum(diffs < 0)
        old_mon = np.abs(pos_steps - neg_steps) / (m - 1)
        
        # 3. Modified Monotonicity (Invoked via external/provided function)
        # Using 10 measurements prior to FPT to estimate noise sigma
        pre_fpt_data = hi_signal[max(0, fpt_index - 10):fpt_index]
        sigma_noise = np.std(pre_fpt_data) if len(pre_fpt_data) > 1 else 1e-3
        mod_mon = mod_monotonicity_func(degradation_phase, sigma=sigma_noise)
        
        # 4. Robustness
        kernel_size = min(4, m)
        kernel = np.ones(kernel_size) / kernel_size
        smoothed_hi = np.convolve(degradation_phase, kernel, mode='same')
        
        # Avoid division by zero by adding small epsilon
        epsilon = 1e-8
        robustness_arr = np.exp(-np.abs(degradation_phase - smoothed_hi) / (degradation_phase + epsilon))
        robustness = np.mean(robustness_arr)
        
        return {
            "pearson": abs(pearson_corr) if not np.isnan(pearson_corr) else 0.0,
            "spearman": abs(spearman_corr) if not np.isnan(spearman_corr) else 0.0,
            "old_mon": old_mon,
            "mod_mon": mod_mon,
            "robustness": robustness
        }

    @staticmethod
    def get_modified_snr(hi_population: List[np.ndarray], fpt_indices: List[int]) -> float:
        """
        Calculates the Modified Signal-to-Noise Ratio (SNR) across a population of units.
        
        Args:
            hi_population (List[np.ndarray]): List of HI arrays for all units in the training set.
            fpt_indices (List[int]): First Prediction Time indices corresponding to each unit.
            
        Returns:
            float: The computed Modified SNR value.
        """
        n_units = len(hi_population)
        if n_units == 0:
            return 0.0
            
        # Extract EOL values and ranges
        eol_values = np.array([hi[-1] for hi in hi_population])
        ranges = np.array([hi[-1] - hi[fpt] for hi, fpt in zip(hi_population, fpt_indices)])
        
        mean_range = np.mean(ranges)
        
        # Compute EOL Variance (theta)
        mean_eol = np.mean(eol_values)
        theta = np.sum((eol_values - mean_eol)**2) / (n_units - 1) if n_units > 1 else 0.0
        
        # Compute residual noise variance (sigma_noise^2)
        total_residual_variance = 0.0
        for hi, fpt in zip(hi_population, fpt_indices):
            deg_phase = hi[fpt:]
            m = len(deg_phase)
            kernel_size = min(4, m)
            if kernel_size > 0:
                smoothed_hi = np.convolve(deg_phase, np.ones(kernel_size)/kernel_size, mode='same')
                total_residual_variance += np.mean((deg_phase - smoothed_hi)**2)
                
        sigma_noise_sq = total_residual_variance / n_units
        
        # Final Modified SNR calculation
        denominator = sigma_noise_sq + theta
        if denominator == 0:
            return 0.0
            
        modified_snr = (mean_range**2) / denominator
        return float(modified_snr)

    def generate_optimal_hi(self, population_features: List[np.ndarray], 
                            fpt_indices: List[int], 
                            get_meta_prob_func: Callable,
                            mod_monotonicity_func: Callable) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Evaluates thousands of weight combinations to generate and select the optimal 
        feature fusion weights based on the multi-objective meta-probability criteria.
        
        Args:
            population_features (List[np.ndarray]): List of 2D arrays (time x features) for each unit.
            fpt_indices (List[int]): FPT indices for each unit.
            get_meta_prob_func (Callable): Function to compute meta-probability (from previous class).
            mod_monotonicity_func (Callable): Function to compute modified monotonicity.
            
        Returns:
            Tuple[np.ndarray, Dict[str, float]]: The optimal weight array and its corresponding metrics.
        """
        num_features = population_features.shape[7]
        n_units = len(population_features)
        
        # 1. Weight Generation
        candidate_weights = self.generate_lhs_weights(num_features, self.num_samples_lhs)
        
        best_score = -np.inf
        optimal_weights = None
        best_metrics_summary = {}

        # 2. Multi-Objective Evaluation
        for weights in candidate_weights:
            # Construct HI for all units using current weights
            hi_population = [np.dot(unit_features, weights) for unit_features in population_features]
            
            spearman_scores = []
            mod_mon_scores = []
            
            for hi, fpt in zip(hi_population, fpt_indices):
                metrics = self.get_metrics(hi, fpt, mod_monotonicity_func)
                spearman_scores.append(metrics["spearman"])
                mod_mon_scores.append(metrics["mod_mon"])
                
            # Modified SNR is a population-level metric, generating 1 value per weight guess
            snr_score = self.get_modified_snr(hi_population, fpt_indices)
            
            # Calculate Meta-Probabilities
            # Assume get_meta_prob_func returns (meta_prob_value, cutoffs, metric_curve)
            meta_prob_spearman, _, _ = get_meta_prob_func(spearman_scores, cutoff_range=(0.8, 1.0))
            meta_prob_mod_mon, _, _ = get_meta_prob_func(mod_mon_scores, cutoff_range=(0.5, 1.0))
            
            # For SNR, typical normalization is applied, simplified here to aggregate performance
            # In a strict setting, SNR can be treated as a single optimization target alongside meta-probs
            
            # Combined Objective Score (Equally weighted maximization scheme)
            combined_objective = meta_prob_spearman + meta_prob_mod_mon + snr_score
            
            if combined_objective > best_score:
                best_score = combined_objective
                optimal_weights = weights
                best_metrics_summary = {
                    "meta_prob_spearman": meta_prob_spearman,
                    "meta_prob_mod_mon": meta_prob_mod_mon,
                    "modified_snr": snr_score,
                    "combined_score": combined_objective
                }
                
        return optimal_weights, best_metrics_summary

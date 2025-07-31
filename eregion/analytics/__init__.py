import numpy as np
from typing import List, Any, Union
from .analytics_base import AnalyticsBase


class EregionAnalytics:
    """
    Analytics engine for neural network monitoring and analysis.
    Provides methods for computing various neural network metrics.
    """
    
    def __init__(self):
        pass
    
    def gradient_norm(self, gradients: List[np.ndarray]) -> float:
        """
        Compute the L2 norm of gradients across all parameters.
        
        Args:
            gradients: List of gradient arrays from model parameters
            
        Returns:
            float: L2 norm of all gradients
        """
        total_norm = 0.0
        for grad in gradients:
            if grad is not None:
                param_norm = np.linalg.norm(grad.flatten())
                total_norm += param_norm ** 2
        return np.sqrt(total_norm)
    
    def entropy_of_predictions(self, outputs: List[Union[np.ndarray, float]]) -> float:
        """
        Compute entropy of model outputs/predictions.
        
        Args:
            outputs: List of output values from model layers
            
        Returns:
            float: Entropy value
        """
        if not outputs:
            return 0.0
        
        # Flatten and normalize outputs
        flat_outputs = []
        for output in outputs:
            if isinstance(output, (list, np.ndarray)):
                flat_outputs.extend(np.array(output).flatten())
            else:
                flat_outputs.append(float(output))
        
        if not flat_outputs:
            return 0.0
        
        # Compute histogram and entropy
        hist, _ = np.histogram(flat_outputs, bins=50, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        if len(hist) == 0:
            return 0.0
        
        return -np.sum(hist * np.log(hist + 1e-10))
    
    def dead_neurons_detection(self, outputs: List[Union[np.ndarray, float]]) -> float:
        """
        Detect dead neurons (neurons with zero or near-zero activation).
        
        Args:
            outputs: List of output values from model layers
            
        Returns:
            float: Percentage of dead neurons
        """
        if not outputs:
            return 0.0
        
        total_neurons = 0
        dead_neurons = 0
        
        for output in outputs:
            if isinstance(output, (list, np.ndarray)):
                output_array = np.array(output).flatten()
                total_neurons += len(output_array)
                dead_neurons += np.sum(np.abs(output_array) < 1e-6)
            else:
                total_neurons += 1
                if abs(float(output)) < 1e-6:
                    dead_neurons += 1
        
        return (dead_neurons / total_neurons * 100) if total_neurons > 0 else 0.0
    
    def layer_activation_distribution(self, outputs: List[Union[np.ndarray, float]]) -> dict:
        """
        Analyze the distribution of activations across layers.
        
        Args:
            outputs: List of output values from model layers
            
        Returns:
            dict: Statistics about activation distributions
        """
        if not outputs:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
        all_activations = []
        for output in outputs:
            if isinstance(output, (list, np.ndarray)):
                all_activations.extend(np.array(output).flatten())
            else:
                all_activations.append(float(output))
        
        if not all_activations:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
        activations = np.array(all_activations)
        return {
            "mean": float(np.mean(activations)),
            "std": float(np.std(activations)),
            "min": float(np.min(activations)),
            "max": float(np.max(activations))
        }


__all__ = ["EregionAnalytics", "AnalyticsBase"] 
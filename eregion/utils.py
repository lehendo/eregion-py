"""
Utility functions for the Eregion neural network analytics library.
"""

import torch
import numpy as np
from typing import Union, Dict, Any, List
import json

# Optional TensorFlow import
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


def validate_model(model: Union[torch.nn.Module, Any]) -> str:
    """
    Validate and determine the framework of a given model.
    
    Args:
        model: PyTorch or TensorFlow model
        
    Returns:
        str: Framework name ('pytorch' or 'tensorflow')
        
    Raises:
        TypeError: If model is not a supported type
    """
    if isinstance(model, torch.nn.Module):
        return "pytorch"
    elif TENSORFLOW_AVAILABLE and isinstance(model, tf.keras.Model):
        return "tensorflow"
    else:
        raise TypeError(
            f"Expected PyTorch (torch.nn.Module) or TensorFlow (tf.keras.Model) model, "
            f"got {type(model)}"
        )


def get_model_info(model: Union[torch.nn.Module, Any]) -> Dict[str, Any]:
    """
    Extract basic information about a model.
    
    Args:
        model: PyTorch or TensorFlow model
        
    Returns:
        dict: Model information including parameters, layers, etc.
    """
    framework = validate_model(model)
    
    if framework == "pytorch":
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        layers = []
        for name, module in model.named_modules():
            if not list(module.children()):  # Leaf modules only
                layers.append({
                    "name": name,
                    "type": module.__class__.__name__,
                    "parameters": sum(p.numel() for p in module.parameters())
                })
        
        return {
            "framework": framework,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "layers": layers
        }
    
    else:  # tensorflow
        total_params = model.count_params()
        trainable_params = sum(
            tf.keras.backend.count_params(w) for w in model.trainable_weights
        )
        
        layers = []
        for layer in model.layers:
            layers.append({
                "name": layer.name,
                "type": layer.__class__.__name__,
                "parameters": layer.count_params()
            })
        
        return {
            "framework": framework,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "layers": layers
        }


def format_metrics(metrics: Dict[str, Any]) -> str:
    """
    Format metrics dictionary into a readable string.
    
    Args:
        metrics: Dictionary of metrics
        
    Returns:
        str: Formatted metrics string
    """
    lines = []
    for key, value in metrics.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for sub_key, sub_value in value.items():
                lines.append(f"  {sub_key}: {sub_value}")
        else:
            lines.append(f"{key}: {value}")
    
    return "\n".join(lines)


def save_metrics_to_file(metrics: Dict[str, Any], filename: str):
    """
    Save metrics to a JSON file.
    
    Args:
        metrics: Dictionary of metrics to save
        filename: Output filename
    """
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)


def load_metrics_from_file(filename: str) -> Dict[str, Any]:
    """
    Load metrics from a JSON file.
    
    Args:
        filename: Input filename
        
    Returns:
        dict: Loaded metrics
    """
    with open(filename, 'r') as f:
        return json.load(f)


def calculate_model_complexity(model: Union[torch.nn.Module, Any]) -> Dict[str, float]:
    """
    Calculate model complexity metrics.
    
    Args:
        model: PyTorch or TensorFlow model
        
    Returns:
        dict: Complexity metrics
    """
    info = get_model_info(model)
    
    return {
        "total_parameters": info["total_parameters"],
        "trainable_parameters": info["trainable_parameters"],
        "parameter_efficiency": info["trainable_parameters"] / info["total_parameters"] if info["total_parameters"] > 0 else 0,
        "layer_count": len(info["layers"])
    }


__all__ = [
    "validate_model",
    "get_model_info", 
    "format_metrics",
    "save_metrics_to_file",
    "load_metrics_from_file",
    "calculate_model_complexity"
]

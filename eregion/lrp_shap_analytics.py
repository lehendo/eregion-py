import numpy as np
import torch
import shap
import tensorflow as tf
from typing import List, Dict
from torch import nn

from zennit.torchvision import ResNetCanonizer # not needed rn
from zennit.composites import EpsilonPlusFlat
from zennit.rules import Epsilon

import innvestigate


class LrpShapAnalytics:

    def __init__(self, model, data_loader=None, framework='pytorch'):
        """
        Initialize a model + data loader.
        """
        self.model = model
        self.data_loader = data_loader
        self.framework = framework
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # For PyTorch

        if self.framework == 'pytorch':
            self.model.to(self.device)
            self.model.eval()

    def compute_lrp_relevance(self) -> Dict[int, np.ndarray]:
        """
        Compute and aggregate LRP relevance scores for each neuron in each layer.
        """
        if self.framework == 'pytorch':
            return self.compute_lrp_pytorch()
        elif self.framework == 'tensorflow':
            return self.compute_lrp_tensorflow()

    def compute_lrp_pytorch(self) -> Dict[int, np.ndarray]:
        """
        LRP for PyTorch models.
        """
        relevance_scores = {}

        for inputs, _ in self.data_loader:
            inputs = inputs.to(self.device)
            relevance_per_layer = self.run_lrp_pytorch(inputs)

            for layer_idx, rel in relevance_per_layer.items():
                if layer_idx not in relevance_scores:
                    relevance_scores[layer_idx] = []
                relevance_scores[layer_idx].append(rel.cpu().detach().numpy())

        for layer_idx, rel_list in relevance_scores.items():
            relevance_scores[layer_idx] = np.mean(rel_list, axis=0)

        return relevance_scores

    def run_lrp_pytorch(self, inputs):
        """
        Back propagation (relevance) for PyTorch models.
        """
        relevance_per_layer = {}
        activations = [inputs]

        x = inputs
        for layer in self.model.children():
            x = layer(x)
            activations.append(x)

        relevance = activations[-1].clone().detach()
        for i, layer in reversed(list(enumerate(self.model.children()))):
            if hasattr(layer, 'weight'):
                weights = layer.weight
                z = torch.matmul(activations[i], weights.T) + 1e-9
                s = relevance / z.sum(dim=1, keepdim=True)
                relevance = torch.matmul(s, weights)
                relevance_per_layer[i] = relevance.clone()

        return relevance_per_layer

    def compute_lrp_tensorflow(self) -> Dict[int, np.ndarray]:
        """
        LRP for TensorFlow models.
        """
        relevance_scores = {}

        for inputs, _ in self.data_loader:
            inputs = tf.convert_to_tensor(inputs)
            relevance_per_layer = self.run_lrp_tensorflow(inputs)

            for layer_idx, rel in relevance_per_layer.items():
                if layer_idx not in relevance_scores:
                    relevance_scores[layer_idx] = []
                relevance_scores[layer_idx].append(rel.numpy())

        for layer_idx, rel_list in relevance_scores.items():
            relevance_scores[layer_idx] = np.mean(rel_list, axis=0)

        return relevance_scores

    def run_lrp_tensorflow(self, inputs):
        """
        Backward propagation (relevance) for TensorFlow models.
        """
        relevance_per_layer = {}
        activations = [inputs]

        x = inputs
        for layer in self.model.layers:
            x = layer(x)
            activations.append(x)

        relevance = activations[-1]
        for i, layer in reversed(list(enumerate(self.model.layers))):
            if isinstance(layer, tf.keras.layers.Dense) or isinstance(layer, tf.keras.layers.Conv2D):
                weights = layer.get_weights()[0]
                z = tf.matmul(activations[i], tf.transpose(weights)) + 1e-9
                s = relevance / tf.reduce_sum(z, axis=1, keepdims=True)
                relevance = tf.matmul(s, weights)
                relevance_per_layer[i] = relevance

        return relevance_per_layer

    def compute_shap_relevance(self) -> Dict[int, np.ndarray]:
        """
        Compute/aggregate SHAP values for each neuron in each layer.
        """
        if self.framework == 'pytorch':
            return self.compute_shap_pytorch()
        elif self.framework == 'tensorflow':
            return self.compute_shap_tensorflow()

    def compute_shap_pytorch(self) -> Dict[int, np.ndarray]:
        """
        Compute SHAP for PyTorch.
        """
        relevance_scores = {}
        baseline = torch.zeros_like(next(iter(self.data_loader))[0])
        num_samples = 50

        for inputs, _ in self.data_loader:
            inputs = inputs.to(self.device)
            shap_values = self.run_shap_pytorch(inputs, baseline, num_samples)

            for layer_idx, shap_val in shap_values.items():
                if layer_idx not in relevance_scores:
                    relevance_scores[layer_idx] = []
                relevance_scores[layer_idx].append(shap_val)

        for layer_idx, shap_list in relevance_scores.items():
            relevance_scores[layer_idx] = np.mean(shap_list, axis=0)

        return relevance_scores

    def run_shap_pytorch(self, inputs, baseline, num_samples):
        """
        Approximate SHAP for PyTorch.
        """
        shap_values = {}

        for i, layer in enumerate(self.model.children()):
            if hasattr(layer, 'weight'):
                contributions = []
                for _ in range(num_samples):
                    subset = torch.randperm(inputs.size(1))[:i + 1]
                    perturbed = inputs.clone()
                    perturbed[:, subset] = baseline[:, subset]
                    f_with = layer(perturbed)
                    f_without = layer(inputs)
                    contributions.append((f_with - f_without).mean(dim=0).cpu().numpy())
                shap_values[i] = np.mean(contributions, axis=0)

        return shap_values

    def compute_shap_tensorflow(self) -> Dict[int, np.ndarray]:
        """
        Compute SHAP for TensorFlow models.
        """
        relevance_scores = {}
        baseline = tf.zeros_like(next(iter(self.data_loader))[0])
        num_samples = 50

        for inputs, _ in self.data_loader:
            inputs = tf.convert_to_tensor(inputs)
            shap_values = self.run_shap_tensorflow(inputs, baseline, num_samples)

            for layer_idx, shap_val in shap_values.items():
                if layer_idx not in relevance_scores:
                    relevance_scores[layer_idx] = []
                relevance_scores[layer_idx].append(shap_val)

        for layer_idx, shap_list in relevance_scores.items():
            relevance_scores[layer_idx] = np.mean(shap_list, axis=0)

        return relevance_scores

    def normalize_scores(self, relevance_dict: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Normalize relevance scores to [0, 1] - meant for visualization.
        """
        normalized_relevance = {}
        for layer_idx, relevance in relevance_dict.items():
            min_val, max_val = relevance.min(), relevance.max()
            normalized_relevance[layer_idx] = (relevance - min_val) / (max_val - min_val)

        return normalized_relevance

    def combine_lrp_shap_scores(self, lrp_relevance: Dict[int, np.ndarray], shap_relevance: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Combine LRP and SHAP relevance scores.
        """
        impact_scores = {}

        for layer_idx in lrp_relevance.keys():
            combined_score = (lrp_relevance[layer_idx] + shap_relevance[layer_idx]) / 2 #i can make this weighted - gotta do research
            impact_scores[layer_idx] = combined_score

        return impact_scores

    def prepare_visualization_data(self) -> Dict[str, Dict[int, np.ndarray]]:
        """
        Prepare data for visualization.
        """
        print("Computing LRP relevance...")
        lrp_relevance = self.compute_lrp_relevance()
        print("Computing SHAP relevance...")
        shap_relevance = self.compute_shap_relevance()

        print("Normalizing relevance scores...")
        normalized_lrp = self.normalize_scores(lrp_relevance)
        normalized_shap = self.normalize_scores(shap_relevance)

        print("Computing combined relevance...")
        impact_scores = self.combine_lrp_shap_scores(lrp_relevance, shap_relevance)

        visualization_data = {
            'lrp': normalized_lrp,
            'shap': normalized_shap,
            'impact': impact_scores
        }

        return visualization_data

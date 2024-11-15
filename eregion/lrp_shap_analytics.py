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
        Compute/aggregate LRP relevance scores for each neuron in each layer.
        """
        if self.framework == 'pytorch':
            return self.compute_lrp_pytorch()
        elif self.framework == 'tensorflow':
            return self.compute_lrp_tensorflow()

    def compute_lrp_pytorch(self) -> Dict[int, np.ndarray]:
        """
        Compute LRP relevance for PyTorch models using Zennit.
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
        Run LRP on the model for given inputs using Zennit.
        """
        relevance_per_layer = {}

        composite = EpsilonPlusFlat(Epsilon())  # Standard LRP rule for stability

        with composite.context(self.model) as modified_model:
            # get outputs
            outputs = modified_model(inputs)
            # Propagate relevance
            relevance = modified_model.relprop(outputs)

        # Gathering relevance scores every layer
        for i, layer in enumerate(self.model.children()):
            if hasattr(layer, 'weight'):
                relevance_per_layer[i] = relevance[layer].detach()

        return relevance_per_layer

    def compute_lrp_tensorflow(self) -> Dict[int, np.ndarray]:
        """
        Compute LRP for TensorFlow models using innvestigate.
        """
        relevance_scores = {}

        analyzer = innvestigate.create_analyzer("lrp.epsilon", self.model)

        for inputs, _ in self.data_loader:
            relevance_per_layer = self.run_lrp_tensorflow(analyzer, inputs)

            for layer_idx, rel in relevance_per_layer.items():
                if layer_idx not in relevance_scores:
                    relevance_scores[layer_idx] = []
                relevance_scores[layer_idx].append(rel.numpy())

        for layer_idx, rel_list in relevance_scores.items():
            relevance_scores[layer_idx] = np.mean(rel_list, axis=0)

        return relevance_scores

    def run_lrp_tensorflow(self, analyzer, inputs):
        """
        Run LRP on the model for given inputs using innvestigate.
        """
        relevance_per_layer = {}

        # get relevance scores
        relevance = analyzer.analyze(inputs)

        # Get relevance for each layer
        for i, layer in enumerate(self.model.layers):
            if 'dense' in layer.name or 'conv' in layer.name:
                relevance_per_layer[i] = relevance[layer.output]

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
        Compute SHAP values for PyTorch.
        """
        relevance_scores = {}

        explainer = shap.DeepExplainer(self.model, next(iter(self.data_loader))[0].to(self.device))

        for inputs, _ in self.data_loader:
            inputs = inputs.to(self.device)
            shap_values = explainer.shap_values(inputs)

            # returns a list of arrays with each corresponding to an output neuron
            for layer_idx, shap_val in enumerate(shap_values):
                if layer_idx not in relevance_scores:
                    relevance_scores[layer_idx] = []
                relevance_scores[layer_idx].append(shap_val.mean(axis=0))

        # Average SHAP values
        for layer_idx, shap_list in relevance_scores.items():
            relevance_scores[layer_idx] = np.mean(shap_list, axis=0)

        return relevance_scores

    def compute_shap_tensorflow(self) -> Dict[int, np.ndarray]:
        """
        Compute SHAP values for TensorFlow using DeepExplainer.
        """
        relevance_scores = {}

        # SHAP requires a background dataset to initialize DeepExplainer
        # Assume `self.data_loader` is an iterable, so we'll take a subset as background
        background_data = next(iter(self.data_loader))[:100]  # Use the first 100 samples as background data

        # Init DeepExplainer
        explainer = shap.DeepExplainer(self.model, background_data)

        for inputs in self.data_loader:
            shap_values = explainer.shap_values(inputs)

            # returns a list of arrays with each corresponding to an output neuron
            for layer_idx, shap_val in enumerate(shap_values):
                if layer_idx not in relevance_scores:
                    relevance_scores[layer_idx] = []
                relevance_scores[layer_idx].append(shap_val.mean(axis=0))

        # Average SHAP values
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

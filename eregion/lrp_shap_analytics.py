import numpy as np
import torch
import shap
import tensorflow as tf
from typing import List, Dict
from torch import nn


class NeuralNetworkAnalytics:
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

    
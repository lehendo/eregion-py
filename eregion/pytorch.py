import torch
from .base import Eregion
from .analytics import EregionAnalytics
from typing import Any


class EregionPyTorch(Eregion):
    def __init__(self, model: torch.nn.Module, name: str, API_KEY: str, auto_track: bool = False,
                 reset: bool = True):
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Expected a PyTorch model (torch.nn.Module), but got {}".format(type(model)))

        super().__init__(name, API_KEY, reset=reset)
        self.model = model
        self.auto_track = auto_track
        self.data_buffer = []
        self.analytics = EregionAnalytics()

        if self.auto_track:
            self._start_auto_tracking()

    def _start_auto_tracking(self):
        """
        Auto-tracking for PyTorch using forward hooks.
        Pushes data every time a forward pass is made.
        """

        def forward_hook(module: torch.nn.Module, input: Any, output: torch.Tensor):
            metrics = {
                'layer': module.__class__.__name__,
                'output': output.detach().cpu().numpy().tolist(),
            }
            self.data_buffer.append(metrics)

            # Push the metrics immediately after each forward pass
            self.push(self._prepare_metrics())

        # Register forward hook on all layers of the model
        for layer in self.model.modules():
            layer.register_forward_hook(forward_hook)

    def _prepare_metrics(self):
        """
        Prepare all collected metrics, including custom ones, for pushing.
        """
        # Compute custom metrics: gradient norm, entropy, dead neurons, etc.
        grad_norm = self.analytics.gradient_norm(
            [param.grad.detach().cpu().numpy() for param in self.model.parameters() if param.grad is not None]
        )
        entropy = self.analytics.entropy_of_predictions(self.data_buffer)
        dead_neurons = self.analytics.dead_neurons_detection(self.data_buffer)
        layer_activation_dist = self.analytics.layer_activation_distribution(self.data_buffer)

        # Return the complete data to be pushed
        return {
            'gradient_norm': grad_norm,
            'entropy': entropy,
            'dead_neurons': dead_neurons,
            'layer_activation_distribution': layer_activation_dist
        }

    def push(self, data=None):
        """
        Push tracked data to the API whenever there is an update.
        """
        if not self.network_id:
            raise Exception("Network ID not found. Make sure the model exists or was created successfully.")

        if self.data_buffer or data:
            # Ensure that the data is a dictionary, not an array
            if data:
                self.data_buffer.append(data)

            # Combine the data_buffer list into a single dictionary
            combined_data = {'metrics': self.data_buffer}  # Wrap the data_buffer in a dictionary

            super().push(combined_data)  # Pass the combined dictionary
            self.data_buffer = []  # Clear buffer after pushing
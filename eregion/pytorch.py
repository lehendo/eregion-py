import torch
from .base import Eregion
from .analytics import EregionAnalytics
from typing import Any


class EregionPyTorch(Eregion):
    def __init__(self, model: torch.nn.Module, name: str, API_KEY: str, auto_track: bool = False, reset: bool = True):
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Expected a PyTorch model (torch.nn.Module), but got {type(model)}")

        super().__init__(name, API_KEY, reset=reset)
        self.model = model
        self.auto_track = auto_track
        self.data_buffer = []
        self.analytics = EregionAnalytics()

        if self.auto_track:
            self._start_auto_tracking()

    def _start_auto_tracking(self):
        """Auto-tracking using forward hooks on model layers for every forward pass."""

        def forward_hook(module, input, output):
            metrics = {
                'layer': module.__class__.__name__,
                'output': output.detach().cpu().numpy().tolist() if isinstance(output, torch.Tensor) else output,
            }
            self.data_buffer.append(metrics)

        # Register forward hook on all leaf layers of the model
        hook_count = 0
        for name, layer in self.model.named_modules():
            if not list(layer.children()):
                layer.register_forward_hook(forward_hook)
                hook_count += 1

        print(f"Auto-tracking started. Hooks registered on {hook_count} modules.")

    def _prepare_metrics(self):
        """Compute custom metrics, including gradient norm, entropy, dead neurons, and activations."""

        # Calculate gradient norm if gradients are available
        grad_norm = (self.analytics.gradient_norm(
            [param.grad.detach().cpu().numpy() for param in self.model.parameters() if param.grad is not None]
        ) if any(param.grad is not None for param in self.model.parameters()) else None)

        # Collect outputs from tracked layers for additional analytics
        outputs = [item['output'] for item in self.data_buffer if 'output' in item]

        return {
            'gradient_norm': grad_norm,
            'entropy': self.analytics.entropy_of_predictions(outputs),
            'dead_neurons': self.analytics.dead_neurons_detection(outputs),
            'layer_activation_distribution': self.analytics.layer_activation_distribution(outputs)
        }

    def push(self, data=None):
        """Push tracked data to the API and clear buffer post-push."""

        if not self.network_id:
            raise Exception("Network ID not found. Ensure the model was created successfully.")

        if data:
            self.data_buffer.append(data)

        if self.data_buffer:
            combined_data = {'metrics': self.data_buffer}

            super().push(combined_data)  # Push combined data
            self.data_buffer.clear()  # Clear buffer after push

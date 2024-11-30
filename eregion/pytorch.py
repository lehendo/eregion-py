import torch
from .base import Eregion
from .analytics import EregionAnalytics
from typing import Any
import numpy as np


# TODO: Make lighter and confirm that model gradients work
class EregionPyTorch(Eregion):
    def __init__(
        self,
        model: torch.nn.Module,
        name: str,
        API_KEY: str,
        auto_track: bool = False,
        reset: bool = True,
    ):
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"Expected a PyTorch model (torch.nn.Module), but got {type(model)}"
            )

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
            if isinstance(output, torch.Tensor):
                metrics = {
                    "layer": module.__class__.__name__,
                    "output_shape": list(output.shape),
                    "output_mean": output.mean().item(),
                    "output_std": output.std().item(),
                }
            else:
                metrics = {
                    "layer": module.__class__.__name__,
                    "output": "Non-tensor output",
                }
            self.data_buffer.append(metrics)

        hook_count = 0
        for name, layer in self.model.named_modules():
            if not list(layer.children()):
                layer.register_forward_hook(forward_hook)
                hook_count += 1

        print(f"Auto-tracking started. Hooks registered on {hook_count} modules.")

    def _prepare_metrics(self):
        """Compute custom metrics, including gradient norm, entropy, dead neurons, and activations."""

        metrics = {}

        # Calculate gradient norm if gradients are available
        if any(param.grad is not None for param in self.model.parameters()):
            grad_norm = self.analytics.gradient_norm(
                [
                    param.grad.detach().cpu().numpy()
                    for param in self.model.parameters()
                    if param.grad is not None
                ]
            )
            metrics["gradient_norm"] = grad_norm

        outputs = [
            item["output_mean"] for item in self.data_buffer if "output_mean" in item
        ]

        outputs = [np.array(o) if isinstance(o, list) else o for o in outputs]

        metrics.update(
            {
                "entropy": self.analytics.entropy_of_predictions(outputs).tolist(),
                "dead_neurons": self.analytics.dead_neurons_detection(outputs),
                "layer_activation_distribution": self.analytics.layer_activation_distribution(
                    outputs
                ),
            }
        )

        return metrics

    def push(self, data=None):
        """Push tracked data to the API and clear buffer post-push."""

        if not self.network_id:
            raise Exception(
                "Network ID not found. Ensure the model was created successfully."
            )

        if data:
            self.data_buffer.append(data)

        if self.data_buffer:
            metrics = self._prepare_metrics()
            combined_data = {"metrics": metrics, "data": self.data_buffer}

            super().push(combined_data)  # Push combined data
            self.data_buffer.clear()  # Clear buffer after push

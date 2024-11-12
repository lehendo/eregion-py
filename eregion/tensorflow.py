import tensorflow as tf
from .base import Eregion
from .analytics import EregionAnalytics
from typing import Any


class EregionTensorFlow(Eregion):
    def __init__(self, model: tf.keras.Model, name: str, API_KEY: str, auto_track: bool = False, reset: bool = True):
        if not isinstance(model, tf.keras.Model):
            raise TypeError(f"Expected a TensorFlow model (tf.keras.Model), but got {type(model)}")

        super().__init__(name, API_KEY, reset=reset)
        self.model = model
        self.auto_track = auto_track
        self.data_buffer = []
        self.analytics = EregionAnalytics()

        if self.auto_track:
            self._start_auto_tracking()

    def _start_auto_tracking(self):
        """Auto-tracking using custom layers for every forward pass."""

        class TrackingLayer(tf.keras.layers.Layer):
            def __init__(self, tracker, **kwargs):
                super().__init__(**kwargs)
                self.tracker = tracker

            def call(self, inputs):
                metrics = {
                    'layer': self.__class__.__name__,
                    'output': tf.nest.map_structure(
                        lambda x: x.numpy().tolist() if isinstance(x, tf.Tensor) and hasattr(x, 'numpy') else x,
                        inputs),
                }
                self.tracker.data_buffer.append(metrics)
                return inputs

        new_layers = []
        for layer in self.model.layers:
            new_layers.append(layer)
            new_layers.append(TrackingLayer(self))

        self.model = tf.keras.Sequential(new_layers)
        print(f"Auto-tracking started. Tracking layers added to {len(new_layers) // 2} modules.")

    def _prepare_metrics(self):
        """Compute custom metrics, including gradient norm, entropy, dead neurons, and activations."""

        # Calculate gradient norm if gradients are available
        gradients = self.model.optimizer.get_gradients(self.model.total_loss, self.model.trainable_variables)
        grad_norm = self.analytics.gradient_norm([g.numpy() for g in gradients if g is not None])

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
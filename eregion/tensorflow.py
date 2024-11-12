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
                    'output_shape': tf.nest.map_structure(lambda x: x.shape.as_list() if hasattr(x, 'shape') else None,
                                                          inputs),
                    'output_mean': tf.nest.map_structure(
                        lambda x: tf.reduce_mean(x) if isinstance(x, tf.Tensor) else None, inputs),
                    'output_std': tf.nest.map_structure(
                        lambda x: tf.math.reduce_std(x) if isinstance(x, tf.Tensor) else None, inputs),
                }
                self.tracker.data_buffer.append(metrics)
                return inputs

        if isinstance(self.model, tf.keras.Sequential):
            new_layers = []
            for layer in self.model.layers:
                new_layers.append(layer)
                new_layers.append(TrackingLayer(self))
            self.model = tf.keras.Sequential(new_layers)
        else:
            # For non-Sequential models - note to self add a more rigorous test
            original_call = self.model.call

            def new_call(inputs, training=None, mask=None):
                outputs = original_call(inputs, training=training, mask=mask)
                TrackingLayer(self)(outputs)
                return outputs

            self.model.call = new_call

        print(f"Auto-tracking started for model of type {type(self.model).__name__}.")

    def _prepare_metrics(self):
        """Compute custom metrics, including entropy, dead neurons, and activations."""
        metrics = {}

        try:
            outputs = []
            for item in self.data_buffer:
                if 'output_mean' in item:
                    try:
                        output = tf.keras.backend.get_value(item['output_mean'])
                    except:
                        output = item['output_mean']
                    outputs.append(output)

            concrete_outputs = [tf.keras.backend.get_value(output) if tf.is_tensor(output) else output for output in
                                outputs]

            metrics.update({
                'entropy': self.analytics.entropy_of_predictions(concrete_outputs),
                'dead_neurons': self.analytics.dead_neurons_detection(concrete_outputs),
                'layer_activation_distribution': self.analytics.layer_activation_distribution(concrete_outputs)
            })
        except Exception as e:
            print(f"Error in preparing metrics: {e}")

        return metrics

    def push(self, data=None):
        """Push tracked data to the API and clear buffer post-push."""
        if not self.network_id:
            raise Exception("Network ID not found. Ensure the model was created successfully.")

        if data:
            self.data_buffer.append(data)

        if self.data_buffer:
            try:
                metrics = self._prepare_metrics()
                combined_data = {'metrics': metrics, 'data': self.data_buffer}

                super().push(combined_data)  # Push combined data
                self.data_buffer.clear()  # Clear buffer after push
            except Exception as e:
                print(f"Error during push operation: {e}")
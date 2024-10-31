import tensorflow as tf
from .base import Eregion
from .analytics import EregionAnalytics
from typing import Any


class EregionTensorFlow(Eregion):
    def __init__(self, model: tf.keras.Model, name: str, API_KEY: str, auto_track: bool = False, reset: bool = True):
        if not isinstance(model, tf.keras.Model):
            raise TypeError("Expected a TensorFlow model (tf.keras.Model), but got {}".format(type(model)))

        super().__init__(name, API_KEY, reset=reset)
        self.model = model
        self.auto_track = auto_track
        self.data_buffer = []
        self.analytics = EregionAnalytics()

        if self.auto_track:
            self._start_auto_tracking()

    def _start_auto_tracking(self):
        """
        Auto-tracking for TensorFlow using custom training steps.
        Pushes data after every training step update.
        """
        original_train_step = self.model.train_step

        @tf.function
        def _start_auto_tracking(self):
            """
            Auto-tracking for TensorFlow using custom training steps.
            Pushes data after every training step update.
            """
            original_train_step = self.model.train_step

            @tf.function
            def custom_train_step(data: Any):
                result = original_train_step(data)

                # We can now collect metrics at the end of the step in eager mode
                tf.keras.backend.set_learning_phase(False) # Ensures we're in inference mode when gathering metrics

                # Collect metrics: evaluate results after the training step
                metrics = {}
                for metric in self.model.metrics:
                    # Evaluate the result if it's a symbolic tensor
                    metrics[metric.name] = metric.result().numpy()
                self.data_buffer.append(metrics)

                # Collect custom metrics like gradient norm, entropy, and dead neurons detection
                grad_norm = self.analytics.gradient_norm([param.numpy() for param in self.model.trainable_variables])
                entropy = self.analytics.entropy_of_predictions(self.data_buffer)
                dead_neurons = self.analytics.dead_neurons_detection(self.data_buffer)
                self.data_buffer.append({
                    'gradient_norm': grad_norm,
                    'entropy': entropy,
                    'dead_neurons': dead_neurons,
                    'layer_activation_distribution': self.analytics.layer_activation_distribution(self.data_buffer)
                })

                # Push the collected data after every training step
                self.push()

                return result

            # Override the train step method to include auto-pushing
            self.model.train_step = custom_train_step

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
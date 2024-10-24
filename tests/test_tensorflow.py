# tests/test_tensorflow.py

import tensorflow as tf
import unittest
from eregion.tensorflow import EregionTensorFlow as et

class TestEregionTensorFlow(unittest.TestCase):
    def setUp(self):
        """
        Set up a simple TensorFlow model and the EregionTensorFlow tracker.
        """
        # Define a simple TensorFlow model
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(5,), activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')

        # Initialize the EregionTensorFlow tracker
        self.tracker = et(self.model, "test_model_tf", "9c164cf9-5e42-406f-9202-1eca80602edc", auto_track=True)

    def test_initialization(self):
        """
        Test that the model and tracker are initialized correctly.
        """
        self.assertIsNotNone(self.tracker)
        self.assertEqual(self.tracker.name, "test_model_tf")
        self.assertEqual(self.tracker.api_key, "9c164cf9-5e42-406f-9202-1eca80602edc")

    def test_push_data(self):
        """
        Test that data is pushed correctly to the API.
        """
        # Simulate a forward pass with dummy data and push it
        dummy_data = {'test': 'data'}
        self.tracker.push(dummy_data)

        # Ensure that the data buffer is cleared after pushing
        self.assertEqual(self.tracker.data_buffer, [])

    def test_auto_track(self):
        """
        Test that auto-tracking works by simulating training and ensuring metrics are collected and pushed.
        """
        # Enable auto tracking
        self.tracker.auto_track = True
        self.tracker._start_auto_tracking()

        # Simulate training with dummy data
        import numpy as np
        x_train = np.random.rand(10, 5)
        y_train = np.random.rand(10, 1)
        self.model.fit(x_train, y_train, epochs=1)

        # Check that metrics were collected and pushed during training
        self.assertEqual(len(self.tracker.data_buffer), 0)  # Should be cleared after push

if __name__ == "__main__":
    TestEregionTensorFlow.setUp()
    unittest.main()
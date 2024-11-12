import tensorflow as tf
import unittest
from eregion.tensorflow import EregionTensorFlow as et
from unittest.mock import patch, MagicMock
import numpy as np

class TestEregionTensorFlow(unittest.TestCase):
    @patch('eregion.base.requests.get')
    @patch('eregion.base.requests.post')
    def setUp(self, mock_post, mock_get):
        mock_get.return_value = MagicMock(status_code=404)
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {'id': 'test_id'})

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(5,), activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        self.tracker = et(self.model, "test_model_tf", "9c164cf9-5e42-406f-9202-1eca80602edc", auto_track=False)

    def test_initialization(self):
        self.assertIsNotNone(self.tracker)
        self.assertEqual(self.tracker.name, "test_model_tf")
        self.assertEqual(self.tracker.api_key, "9c164cf9-5e42-406f-9202-1eca80602edc")
        self.assertFalse(self.tracker.auto_track)

    @patch('eregion.base.requests.post')
    def test_push_data(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        self.tracker.push({'test': 'data'})
        self.assertEqual(self.tracker.data_buffer, [])
        mock_post.assert_called_once()

    @patch('eregion.base.requests.post')
    def test_auto_track(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)

        # Enable auto-tracking and verify it's activated
        self.tracker.auto_track = True
        self.tracker._start_auto_tracking()
        self.assertTrue(self.tracker.auto_track, "Auto-tracking should be enabled.")

        self.tracker.model.build((None, 5))

        x_train = np.random.rand(10, 5)
        _ = self.tracker.model(x_train)

        self.assertGreater(len(self.tracker.data_buffer), 0,
                           "After forward pass, data buffer should contain tracking data.")

        self.tracker.push()
        mock_post.assert_called()

        self.assertEqual(len(self.tracker.data_buffer), 0, "After push is called, data buffer should be empty.")

    def test_invalid_model_type(self):
        with self.assertRaises(TypeError):
            et("not a model", "test", "api_key")


if __name__ == "__main__":
    unittest.main()
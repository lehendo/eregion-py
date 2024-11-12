import torch
import unittest
from eregion.pytorch import EregionPyTorch as ep
import torch.nn as nn
from unittest.mock import patch

class TestEregionPyTorch(unittest.TestCase):
    def setUp(self):
        self.model = nn.Linear(5, 3)
        self.tracker = ep(self.model, "test_model", "9c164cf9-5e42-406f-9202-1eca80602edc", auto_track=False)

    def test_initialization(self):
        self.assertIsNotNone(self.tracker)

    def test_push_data(self):
        self.tracker.push({'test': 'data'})
        self.assertEqual(self.tracker.data_buffer, [])

    def test_auto_track(self):
        # Enable auto-tracking and verify it's activated
        self.tracker.auto_track = True
        self.tracker._start_auto_tracking()
        self.assertTrue(self.tracker.auto_track, "Auto-tracking should be enabled.")

        x_train = torch.rand(10, 5)
        _ = self.tracker.model(x_train)

        self.assertGreater(len(self.tracker.data_buffer), 0, "After forward pass, data buffer should contain tracking data.")

        with patch.object(self.tracker, 'push', wraps=self.tracker.push) as mock_push:
            self.tracker.push()
            mock_push.assert_called_once()

        self.assertEqual(len(self.tracker.data_buffer), 0, "After push is called, data buffer should be empty.")

    def test_invalid_model_type(self):
        with self.assertRaises(TypeError):
            ep("not a model", "test", "api_key")

    def test_prepare_metrics(self):
        # Enable auto-tracking
        self.tracker.auto_track = True
        self.tracker._start_auto_tracking()

        x_train = torch.rand(10, 5)
        _ = self.tracker.model(x_train)

        metrics = self.tracker._prepare_metrics()

        self.assertIn('entropy', metrics)
        self.assertIn('dead_neurons', metrics)
        self.assertIn('layer_activation_distribution', metrics)
        # Note: gradient_norm might not be present if the model hasn't been trained

        for item in self.tracker.data_buffer:
            self.assertIn('layer', item)
            self.assertIn('output_shape', item)
            self.assertIn('output_mean', item)
            self.assertIn('output_std', item)

if __name__ == "__main__":
    unittest.main()

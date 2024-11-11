import torch
import unittest
from eregion.pytorch import EregionPyTorch as ep
import numpy as np
import torch.nn as nn

class TestEregionPyTorch(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Linear(10, 5)
        self.tracker = ep(self.model, "test_model", "9c164cf9-5e42-406f-9202-1eca80602edc", auto_track=False)

    def test_initialization(self):
        self.assertIsNotNone(self.tracker)

    def test_push_data(self):
        self.tracker.push({'test': 'data'})
        self.assertEqual(self.tracker.data_buffer, [])

    def test_auto_track(self):
        self.tracker.auto_track = True
        self.tracker._start_auto_tracking()

        x_train = torch.rand(10, 5)
        y_train = torch.rand(10, 1)

        model = nn.Linear(5, 1)
        criterion = nn.MSELoss()

        model.train()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                param -= 0.01 * param.grad

        self.assertEqual(len(self.tracker.data_buffer), 0)

if __name__ == "__main__":
    unittest.main()
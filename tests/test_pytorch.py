import torch
import unittest
from eregion.pytorch import EregionPyTorch as ep

class TestEregionPyTorch(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Linear(10, 5)
        self.tracker = ep(self.model, "test_model", "9c164cf9-5e42-406f-9202-1eca80602edc", auto_track=False)

    def test_initialization(self):
        self.assertIsNotNone(self.tracker)

    def test_push_data(self):
        self.tracker.push({'test': 'data'})
        self.assertEqual(self.tracker.data_buffer, [])

if __name__ == "__main__":
    unittest.main()
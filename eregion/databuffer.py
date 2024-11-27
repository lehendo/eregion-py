from typing import List


class DataBuffer:
    def __init__(self):
        """
        Initialize DataBuffer with dictionary of all hook's data being logged per training iteration.
        """
        self.buffer = {
            "outputs": [],
            "activations": [],
            "gradients": [],
            "labels": [],
            "predictions": [],
        }

    def get(self, metric: str) -> List:
        """
        Obtain the data of a specific metric.
        """
        try:
            return self.buffer[metric]
        except KeyError:
            raise Exception(f"The metric '{metric}' was not found in the data buffer.")

    def update(self, metric: str, value):
        """
        Update or create new metric in DataBuffer.
        (Later, this will be integral in adding multimodal features, etc. based off hooks in PyTorch, TF, etc. implementations)
        """
        if metric in self.buffer:
            self.buffer[metric].append(value)
            return

        self.buffer[metric] = [value]

    def reset(self):
        """
        Conduct a hard reset, clearing out the DataBuffer altogether.
        """
        self.buffer = {
            "outputs": [],
            "activations": [],
            "gradients": [],
            "labels": [],
            "predictions": [],
        }

    def clear(self, metric: str):
        """
        Refreshing a specific metric
        """
        self.buffer[metric] = []

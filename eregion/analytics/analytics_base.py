from abc import ABC, abstractmethod
from typing import Any, Union
import tensorflow as tf
import torch
from .databuffer import DataBuffer

ModelType = Union[torch.nn.Module | tf.keras.Model]
DataLoaderType = Union[torch.utils.data.DataLoader, tf.data.Dataset]


class AnalyticsBase(ABC):
    """
    Base class for analytics, ensuring consistent return data formats.
    All any analytics submodule needs to do is have a compute() function.

    :param model (Any): The neural network model being provided
    :param data_loader (Any): The data loader being used
    :param framework (str): The framework being used
                            (either 'pytorch' or 'tensorflow')
    :param data_buffer (DataBuffer): The raw, layer-by-layer data used for some
                                     analytics
    """

    def __init__(
        self,
        model: ModelType,
        data_loader: DataLoaderType,
        framework: str,
        data_buffer: DataBuffer,
    ):
        self.model = model
        self.data_loader = data_loader
        self.framework = framework
        self.data_buffer = data_buffer

        if self.framework not in ["pytorch", "tensorflow"]:
            raise TypeError(
                f"Expected either a PyTorch model (torch.nn.Module) or Tensorflow model (tf.keras.Model) but got {framework}"
            )

    @abstractmethod
    def compute(self) -> dict[str, Any]:
        """
        Compute analytics and return results in a standard format. All
        analytics are effectively returned in dict[str, Any] format to be
        easily added to DataBuffer.
        """
        pass

    def which_framework(self):
        return self.framework

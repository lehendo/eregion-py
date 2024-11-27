from abc import ABC, abstractmethod
from typing import Dict, Any


class AnalyticsBase(ABC):
    """
    Base class for analytics submodule, ensuring consistent return data formats.
    All any analytics submodule needs to do is have a compute() function.

    :param model (Any): The neural network model being provided
    :param data_loader (Any): The data loader being used
    :param framework (str): The framework being used
                            (either 'pytorch' or 'tensorflow')
    """

    def __init__(self, model: Any, data_loader: Any, framework: str):
        self.model = model
        self.data_loader = data_loader
        self.framework = framework

    @abstractmethod
    def compute(self) -> Dict[str, Any]:
        """
        Compute analytics and return results in a standard format. All
        analytics are effectively returned in Dict[str, Any] format to be
        easily added to DataBuffer.
        """
        pass

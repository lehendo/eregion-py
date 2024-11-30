class DataBuffer:
    def __init__(self):
        """
        Initialize DataBuffer with dictionary of all hook's data being logged per training iteration.
        """
        self.data = {}

    def add_layer_data(self, epoch: int, layer_name: str, metrics: dict):
        if epoch not in self.data:
            self.data[epoch] = {}
        if layer_name not in self.data[epoch]:
            self.data[epoch][layer_name] = {}
        self.data[epoch][layer_name].update(metrics)

    def add_model_data(self, epoch: int, metrics: dict):
        if epoch not in self.data:
            self.data[epoch] = {}
        if "whole_network" not in self.data[epoch]:
            self.data[epoch]["whole_network"] = {}
        self.data[epoch]["whole_network"].update(metrics)

    def get_layer_data(self, epoch: int, layer_name: str):
        if layer_name in self.data.get(epoch, {}):
            return self.data.get(epoch, {}).get(layer_name, {})
        return self.data.get(epoch, {})

    def get_model_data(self, epoch: int):
        return self.data.get(epoch, {}).get("whole_network", {})

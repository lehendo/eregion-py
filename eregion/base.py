import requests


# TODO: Modify API reqs and check step-by-step (write unittests) after refractor schema
class Eregion:
    API_URL = (
        "https://www.eregion.dev/api/v1/network"  # The current Eregion API endpoint
    )

    def __init__(self, name: str, API_KEY: str, reset: bool = True):
        """
        Base class for interacting with the Eregion backend.

        :param name: Name of the neural network.
        :param API_KEY: API key for authentication.
        :param reset: Whether to reset data if the model already exists.
        :param version: Version of the neural network.
        """

        self.name = name
        self.api_key = API_KEY
        self.network_id = None

        if self._check_model_exists():
            if reset:
                self._reset_model_data()  # Automatically reset model data if reset=True
            else:
                raise Exception(
                    f"Model '{self.name}' already exists. Set reset=True to reset the data."
                )
        else:
            self._create_model()

    def _check_model_exists(self):
        """
        Check if the model with the given name and version exists.
        """
        response = requests.get(
            f"{self.API_URL}?name={self.name}&apiKey={self.api_key}", timeout=20
        )
        if response.status_code == 200:
            network_data = response.json()
            if network_data:
                self.network_id = network_data["id"]
                return True
        return False

    def _create_model(self):
        """
        Create a new model via a POST request.
        """
        response = requests.post(
            self.API_URL,
            json={
                "name": self.name,
                "data": {},
                "action": "CREATE",
                "apiKey": self.api_key,
            },
        )
        if response.status_code == 200:
            network_data = response.json()
            self.network_id = network_data["id"]
        else:
            raise Exception(
                f"Error creating model: {response.json().get('error', 'Unknown error')}"
            )

    def _reset_model_data(self):
        """
        Reset model data via a POST request (HARD_RESET).
        """
        response = requests.post(
            self.API_URL,
            json={
                "name": self.name,
                "data": {},
                "action": "HARD_RESET",
                "apiKey": self.api_key,
            },
        )
        if response.status_code != 200:
            raise Exception(
                f"Error resetting model data: {response.json().get('error', 'Unknown error')}"
            )

    def push_data(self, data):
        """
        Push raw data to the API. This is everything that comes out of DataBuffer.
        """
        response = requests.post(
            self.API_URL,
            json={
                "name": self.name,
                "data": data,
                "action": "ADD_DATA",
                "apiKey": self.api_key,
                "networkId": self.network_id,
            },
        )
        if response.status_code != 200:
            raise Exception(
                f"Error pushing data: {response.json().get('error', 'Unknown error')}"
            )

    def push_metric(self, metric_type: str, value: float):
        """
        Push a single metric to the API. These are specific analytics values.
        """
        response = requests.post(
            f"{self.API_URL}/metrics",
            json={
                "metricType": metric_type,
                "value": value,
                "networkId": self.network_id,
                "apiKey": self.api_key,
            },
        )
        if response.status_code != 200:
            raise Exception(
                f"Error pushing metric: {response.json().get('error', 'Unknown error')}"
            )

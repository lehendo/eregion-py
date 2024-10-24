import requests

class Eregion:
    api_url = "http://localhost:3000/api/v1/network"

    def __init__(self, name: str, API_KEY: str, reset: bool = True):
        self.name = name
        self.api_key = API_KEY
        self.network_id = None

        if self._check_model_exists():
            if reset:
                self._reset_model_data()  # Automatically reset model data if reset=True
            else:
                raise Exception(f"Model '{self.name}' already exists. Set reset=True to reset the data.")
        else:
            self._create_model()

    def _check_model_exists(self):
        """
        Check if the model with the given name exists.
        """
        response = requests.get(f'{self.api_url}?name={self.name}&apiKey={self.api_key}')
        if response.status_code == 200:
            network_data = response.json()
            if network_data:
                self.network_id = network_data['id']
                return True
        return False

    def _create_model(self):
        """
        Create a new model via a POST request.
        """
        response = requests.post(
            self.api_url,
            json={'name': self.name, 'data': {}, 'action': 'CREATE', 'apiKey': self.api_key}
        )
        if response.status_code == 200:
            network_data = response.json()
            self.network_id = network_data['id']
        else:
            raise Exception(f"Error creating model: {response.json()['error']}")

    def _reset_model_data(self):
        """
        Reset model data via a POST request (HARD_RESET).
        """
        response = requests.post(
            self.api_url,
            json={'name': self.name, 'data': {}, 'action': 'HARD_RESET', 'apiKey': self.api_key}
        )
        if response.status_code != 200:
            raise Exception(f"Error resetting model data: {response.json()['error']}")

    def push(self, data):
        """
        Push model data to the API.
        """
        response = requests.post(
            self.api_url,
            json={
                'name': self.name,
                'data': data,
                'action': 'ADD_DATA',
                'apiKey': self.api_key,
                'networkId': self.network_id
            }
        )
        if response.status_code != 200:
            raise Exception(f"Error pushing data: {response.json()['error']}")
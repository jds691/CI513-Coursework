from typing import Any


class PredictionModel:
    def __init__(self, name:str, config: dict[str, Any]) -> None:
        if not config['Disable Cache']:
            config['Model'] = name
            config_hash_code = hash(config)

    def train(self) -> None:
        pass

    def predict(self) -> None:
        pass
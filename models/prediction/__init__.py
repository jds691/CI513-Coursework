from typing import Any

from sklearn.tree import DecisionTreeRegressor

from models.config import ConfigOption, PredictionModelName


class PredictionModel:
    def __init__(self, name:str, config: dict[str, Any]) -> None:
        if not config[ConfigOption.DISABLE_CACHE]:
            config['Model'] = name
            config_hash_code = hash(config)

    def train(self, x: Any, y: Any, validation_data: tuple = None) -> None:
        pass

    def predict(self, x: Any) -> Any:
        pass

class DecisionTreeModel(PredictionModel):

    def __init__(self, config: dict[str, Any]):
        super().__init__(PredictionModelName.DECISION_TREE, config)
        self.model = DecisionTreeRegressor()

    def train(self, x: Any, y: Any, validation_data: tuple = None) -> None:
        self.model = DecisionTreeRegressor().fit(x, y)

    def predict(self, x: Any) -> Any:
        return self.model.predict(x)
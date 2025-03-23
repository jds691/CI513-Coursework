from enum import StrEnum, auto
from typing import Any

from sklearn.tree import DecisionTreeClassifier


class PredictionModelName(StrEnum):
    DECISION_TREE = 'Decision Tree'

class PredictionConfigOption(StrEnum):
    ENABLED_MODELS = 'Enabled Models'
    PROBLEM = 'Problem'
    DISABLE_CACHE = 'Disable Cache'

class PredictionProblem(StrEnum):
    SUPPLY = 'Supply'
    DEMAND = 'Demand'

class PredictionModelDisableCache(StrEnum):
    NO = 'No'
    YES = 'Yes'

class PredictionModel:
    def __init__(self, name:str, config: dict[str, Any]) -> None:
        if not config[PredictionConfigOption.DISABLE_CACHE]:
            config['Model'] = name
            config_hash_code = hash(config)

    def train(self, x: Any, y: Any, validation_data: tuple = None) -> None:
        pass

    def predict(self, x: Any) -> Any:
        pass

class DecisionTreeModel(PredictionModel):

    def __init__(self, config: dict[str, Any]):
        super().__init__(PredictionModelName.DECISION_TREE, config)
        self.model = DecisionTreeClassifier()

    def train(self, x: Any, y: Any, validation_data: tuple = None) -> None:
        self.model = DecisionTreeClassifier().fit(x, y)

    def predict(self, x: Any) -> Any:
        return self.model.predict(x)

class PredictionModelRunner:
    def __init__(self, config: dict) -> None:
        self.config: dict = config

    def run_models(self) -> None:
        models: list[PredictionModel] = self._create_models()

        # TODO: Implement the main processing in here. This is where the main bulk of the processing and visualisations should be

    def _create_model_config(self) -> dict:
        model_config: dict = {
            PredictionConfigOption.DISABLE_CACHE: self.config[PredictionConfigOption.DISABLE_CACHE]
        }

        return model_config

    def _create_models(self) -> list[PredictionModel]:
        models: list[PredictionModel] = []

        for model in self.config[PredictionConfigOption.ENABLED_MODELS]:
            match model:
                case PredictionModelName.DECISION_TREE:
                    models.append(DecisionTreeModel(self._create_model_config()))

        return models
from enum import StrEnum
from typing import Any

from sklearn.tree import DecisionTreeClassifier

from dataset import WeatherDataset, EnergyDataset, MergedDataset


class PredictionModelName(StrEnum):
    DECISION_TREE = 'Decision Tree'

class PredictionConfigOption(StrEnum):
    ENABLED_MODELS = 'Enabled Models'
    PROBLEM = 'Problem'
    DISABLE_CACHE = 'Disable Cache'
    FEATURE_SETS = 'Feature Sets'

class PredictionProblem(StrEnum):
    SUPPLY = 'Supply'
    DEMAND = 'Demand'

class PredictionFeatureSets(StrEnum):
    # These are based on the same feature sets as the existing models
    HISTORICAL_ONLY = 'Historical (Baseline)'
    HISTORICAL_WEATHER = 'Historical + Weather Data (Baseline)'
    ALL_FEATURES = 'Full Feature Set'
    HISTORICAL_TEMPERATURE = 'Historical + Temperature'
    HISTORICAL_WIND_SPEED = 'Historical + Wind Speed'
    HISTORICAL_HUMIDITY = 'Historical + Humidity'
    HISTORICAL_PRESSURE = 'Historical + Pressure'
    HISTORICAL_IRRADIANCE = 'Historical + Irradiance'
    HISTORICAL_CLEARNESS = 'Historical + Clearness'
    HISTORICAL_UV_INDEX = 'Historical + UV Index'
    HISTORICAL_WITH_LAG_FEATURES = 'Historical with Lag Features'
    HISTORICAL_WITH_TIME_OF_DAY_AND_LAG_FEATURES = 'Historical with Time Of Day + Lag Features'

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

        # Adding final dataset features

        weather_dataset = WeatherDataset('data/Sakakah 2021 weather dataset.csv' if self.config[PredictionConfigOption.PROBLEM] == PredictionProblem.SUPPLY else 'data/Sakakah 2021 weather dataset Demand.csv')
        energy_dataset = EnergyDataset('data/Sakakah 2021 PV Supply dataset.xlsx' if self.config[PredictionConfigOption.PROBLEM] == PredictionProblem.SUPPLY else 'data/Sakakah 2021 Demand dataset.xlsx', datetime_column='Date & Time' if self.config[PredictionConfigOption.PROBLEM] == PredictionProblem.SUPPLY else 'DATE-TIME')

        merged_dataset = MergedDataset(energy_dataset, weather_dataset)
        merged_dataset.clean()

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
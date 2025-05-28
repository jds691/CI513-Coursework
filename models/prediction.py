from typing import Any

from keras.src.layers import Bidirectional
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM

from models.config import ConfigOption, PredictionModelName


class PredictionModel:
    require_3d_input = False

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
        self.model.fit(x, y)

    def predict(self, x: Any) -> Any:
        return self.model.predict(x)

class ExtraTreeModel(PredictionModel):

    def __init__(self, config: dict[str, Any]):
        super().__init__(PredictionModelName.EXTRA_TREE, config)
        self.model = ExtraTreesRegressor()

    def train(self, x: Any, y: Any, validation_data: tuple = None) -> None:
        self.model.fit(x, y)

    def predict(self, x: Any) -> Any:
        return self.model.predict(x)

class RandomForestModel(PredictionModel):

    def __init__(self, config: dict[str, Any]):
        super().__init__(PredictionModelName.RANDOM_FOREST, config)
        self.model = RandomForestRegressor()

    def train(self, x: Any, y: Any, validation_data: tuple = None) -> None:
        self.model.fit(x, y)

    def predict(self, x: Any) -> Any:
        return self.model.predict(x)

class LSTMModel(PredictionModel):
    def __init__(self, config: dict[str, Any]):
        self.require_3d_input = True

        super().__init__(PredictionModelName.LSTM, config)
        self.model = Sequential()
        self.model.add(LSTM(50, activation='relu'))
        self.model.add(Dense(1))  # output layer
        self.model.compile(optimizer='adam', loss='mse')

    def train(self, x: Any, y: Any, validation_data: tuple = None) -> None:
        self.model.fit(x, y, epochs=20, batch_size=32, validation_data=validation_data, verbose=1)

    def predict(self, x: Any) -> Any:
        return self.model.predict(x)

class BiLSTMModel(PredictionModel):
    def __init__(self, config: dict[str, Any]):
        self.require_3d_input = True

        super().__init__(PredictionModelName.BILSTM, config)
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(50, activation='relu')))
        self.model.add(Dense(1))  # output layer
        self.model.compile(optimizer='adam', loss='mse')

    def train(self, x: Any, y: Any, validation_data: tuple = None) -> None:
        self.model.fit(x, y, epochs=20, batch_size=32, validation_data=validation_data, verbose=1)

    def predict(self, x: Any) -> Any:
        return self.model.predict(x)
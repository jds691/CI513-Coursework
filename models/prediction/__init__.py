import calendar
from enum import StrEnum
from typing import Any

import numpy as np
from pandas import DataFrame
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor

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
        self.model = DecisionTreeRegressor()

    def train(self, x: Any, y: Any, validation_data: tuple = None) -> None:
        self.model = DecisionTreeRegressor().fit(x, y)

    def predict(self, x: Any) -> Any:
        return self.model.predict(x)

class PredictionModelRunner:
    def __init__(self, config: dict) -> None:
        self.config: dict = config

    def run_models(self) -> None:
        weather_dataset = WeatherDataset('data/Sakakah 2021 weather dataset.csv' if self.config[PredictionConfigOption.PROBLEM] == PredictionProblem.SUPPLY else 'data/Sakakah 2021 weather dataset Demand.csv')
        energy_dataset = EnergyDataset('data/Sakakah 2021 PV Supply dataset.xlsx' if self.config[PredictionConfigOption.PROBLEM] == PredictionProblem.SUPPLY else 'data/Sakakah 2021 Demand dataset.xlsx', datetime_column='Date & Time' if self.config[PredictionConfigOption.PROBLEM] == PredictionProblem.SUPPLY else 'DATE-TIME')

        merged_dataset = MergedDataset(energy_dataset, weather_dataset)
        merged_dataset.clean()

        dataset = merged_dataset.dataset

        results = []

        for feature_set in self.config[PredictionConfigOption.FEATURE_SETS]:
            features: list[str] = self._get_feature_set_columns(feature_set, dataset)

            for model_name in self.config[PredictionConfigOption.ENABLED_MODELS]:
                model = self._create_model(model_name)

                for year in dataset['Year'].unique():

                    monthly_predictions = []
                    monthly_true_values = []
                    full_year_data = dataset[dataset['Year'] == year].sort_values(by='DATE-TIME')

                    for month in dataset['Month'].unique():
                        # Filter data for the current month
                        month_data = dataset[(dataset['Year'] == year) & (dataset['Month'] == month)]

                        # Split 70/30 within the month
                        day_cutoff = int(month_data['Day'].max() * 0.7)
                        train_by_days = month_data[month_data['Day'] <= day_cutoff]
                        test_by_days = month_data[month_data['Day'] > day_cutoff]

                        X_train = train_by_days[features]
                        y_train = train_by_days['MW']
                        y_train = y_train.to_numpy()

                        X_test = test_by_days[features]
                        y_test = test_by_days['MW']
                        y_test = y_test.to_numpy()

                        # Scale the data
                        scaler = MinMaxScaler()
                        #train_scaled = scaler.fit_transform(train_by_days[features])
                        #test_scaled = scaler.transform(test_by_days[features])

                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.fit_transform(X_test)

                        # If 3D inputs are required e.g. LSTM
                        #X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # Add third dimension
                        #X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

                        # if using nsteps
                        #def create_sequences(data, n_steps):
                        #    X, y = [], []
                        #    for i in range(len(data) - n_steps):
                        #        X.append(data[i:i + n_steps, :])
                        #        y.append(data[i + n_steps, 0])  # Assuming the first column is the target 'MW'
                        #    return np.array(X), np.array(y)

                        #n_steps = 24  # 24-hour sequences
                        #X_train, y_train = create_sequences(train_scaled, n_steps)
                        #X_test, y_test = create_sequences(test_scaled, n_steps)

                        model.train(X_train, y_train, validation_data=(X_test, y_test))

                        y_predicted = model.predict(X_test)

                        mse = mean_squared_error(y_test, y_predicted)
                        mae = mean_absolute_error(y_test, y_predicted)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test, y_predicted)

                        # Store the result
                        results.append({
                            'Year': year,
                            'Month': calendar.month_name[month],
                            'Feature Set': feature_set,
                            'MSE': mse,
                            'MAE': mae,
                            'RMSE': rmse,
                            'R2': r2,
                            'Model': model_name,
                        })

                        print(f"Completed {feature_set} on model {model_name} for {calendar.month_name[month]} {year} with "
                              f"MSE: {mse}, MAE: {mae}, RMSE: {rmse}, R-squared: {r2}")

                        # Inverse transform to get back to original scale
                        placeholder = np.zeros((y_test.shape[0], X_train.shape[1]))
                        placeholder[:, 0] = y_test.ravel()  # Assuming y_test is the first column after scaling
                        y_test_original = scaler.inverse_transform(placeholder)[:, 0]

                        placeholder[:, 0] = y_predicted.ravel()
                        y_pred_original = scaler.inverse_transform(placeholder)[:, 0]

                        # Collect data for plotting
                        #monthly_true_values.append((test_by_days['DATE-TIME'][n_steps:], y_test_original))
                        #monthly_predictions.append((test_by_days['DATE-TIME'][n_steps:], y_pred_original))
                        monthly_true_values.append((test_by_days['DATE-TIME'], y_test_original))
                        monthly_predictions.append((test_by_days['DATE-TIME'], y_pred_original))

        self._create_visualisations_from_results()

    def _create_visualisations_from_results(self) -> None:
        pass

    def _create_model_config(self) -> dict:
        model_config: dict = {
            PredictionConfigOption.DISABLE_CACHE: self.config[PredictionConfigOption.DISABLE_CACHE]
        }

        return model_config

    def _create_model(self, model_name: str) -> PredictionModel:
        match model_name:
            case PredictionModelName.DECISION_TREE:
                return DecisionTreeModel(self._create_model_config())

        raise Exception(f'No model for name: \'{model_name}\'')

    def _get_feature_set_columns(self, feature_set: str, dataset: DataFrame) -> list[str]:
        match feature_set:
            case PredictionFeatureSets.HISTORICAL_ONLY:
                return ['MW']
            case PredictionFeatureSets.HISTORICAL_WEATHER:
                return ['MW', 'Temperature', 'Humidity', 'Pressure', 'Wind Speed', 'Wind Direction', 'Downward Irradiance', 'Clearness Index', 'UV Index']
            case PredictionFeatureSets.ALL_FEATURES:
                return [col for col in dataset.columns if col not in ['DATE-TIME', 'Date & Time', 'DATE', 'Date']]
            case PredictionFeatureSets.HISTORICAL_TEMPERATURE:
                return ['MW', 'Temperature']
            case PredictionFeatureSets.HISTORICAL_WIND_SPEED:
                return ['MW', 'Wind Speed']
            case PredictionFeatureSets.HISTORICAL_HUMIDITY:
                return ['MW', 'Humidity']
            case PredictionFeatureSets.HISTORICAL_PRESSURE:
                return ['MW', 'Pressure']
            case PredictionFeatureSets.HISTORICAL_IRRADIANCE:
                return ['MW', 'Downward Irradiance']
            case PredictionFeatureSets.HISTORICAL_CLEARNESS:
                return ['MW', 'Clearness Index']
            case PredictionFeatureSets.HISTORICAL_UV_INDEX:
                return ['MW', 'UV Index']
            case PredictionFeatureSets.HISTORICAL_WITH_LAG_FEATURES:
                return ['MW', 'Lag_1H_MW', 'Lag_24H_MW', 'Lag_1H_Temperature', 'Lag_24H_Temperature']
            case PredictionFeatureSets.HISTORICAL_WITH_TIME_OF_DAY_AND_LAG_FEATURES:
                return ['MW', 'Hour', 'Lag_1H_MW', 'Lag_24H_MW']
            case _:
                raise Exception(f'Unknown feature set: \'{feature_set}\'')

if __name__ == '__main__':
    PredictionModelRunner({
        PredictionConfigOption.ENABLED_MODELS: ['Decision Tree'],
        PredictionConfigOption.DISABLE_CACHE: True,
        PredictionConfigOption.PROBLEM: 'Supply',
        PredictionConfigOption.FEATURE_SETS: ['Full Feature Set'],
    }).run_models()
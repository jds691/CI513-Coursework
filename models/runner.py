import calendar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from dataset import WeatherDataset, EnergyDataset, MergedDataset
from models.config import Problem, FeatureSet
from models.prediction import *


class ModelRunner:
    def __init__(self, config: dict) -> None:
        self.config: dict = config

    def run_models(self) -> None:
        weather_dataset = WeatherDataset('data/Sakakah 2021 weather dataset.csv' if self.config[ConfigOption.PROBLEM] == Problem.SUPPLY else 'data/Sakakah 2021 weather dataset Demand.csv')
        energy_dataset = EnergyDataset('data/Sakakah 2021 PV Supply dataset.xlsx' if self.config[ConfigOption.PROBLEM] == Problem.SUPPLY else 'data/Sakakah 2021 Demand dataset.xlsx', datetime_column='Date & Time' if self.config[ConfigOption.PROBLEM] == Problem.SUPPLY else 'DATE-TIME')

        merged_dataset = MergedDataset(energy_dataset, weather_dataset)
        merged_dataset.clean()

        dataset = merged_dataset.dataset

        results = []

        for feature_set in self.config[ConfigOption.FEATURE_SETS]:
            features: list[str] = self._get_feature_set_columns(feature_set, dataset)

            for model_name in self.config[ConfigOption.ENABLED_MODELS]:
                model = self._create_model(model_name)

                for year in dataset['Year'].unique():
                    for month in dataset['Month'].unique():
                        # Filter data for the current month
                        month_data = dataset[(dataset['Year'] == year) & (dataset['Month'] == month)]

                        # Split 70/30 within the month
                        day_cutoff = int(month_data['Day'].max() * 0.7)
                        train_by_days = month_data[month_data['Day'] <= day_cutoff]
                        test_by_days = month_data[month_data['Day'] > day_cutoff]

                        # Scale the data
                        scaler = MinMaxScaler()

                        train_scaled = scaler.fit_transform(train_by_days[features])
                        test_scaled = scaler.transform(test_by_days[features])

                        def create_sequences(data, n_steps):
                            X, y = [], []
                            for i in range(len(data) - n_steps):
                                if not model.require_3d_input:
                                    X.append(data[i:i + n_steps, :].flatten())
                                else:
                                    X.append(data[i:i + n_steps, :])
                                y.append(data[i + n_steps, 0])  # Assuming the first column is the target 'MW'
                            return np.array(X), np.array(y)

                        n_steps = 24  # 24-hour sequences
                        X_train, y_train = create_sequences(train_scaled, n_steps)
                        X_test, y_test = create_sequences(test_scaled, n_steps)

                        model.train(X_train, y_train, validation_data=(X_test, y_test))

                        y_predicted = model.predict(X_test)

                        # Inverse transform to get back to original scale
                        placeholder = np.zeros((y_test.shape[0], train_scaled.shape[1]))
                        placeholder[:, 0] = y_test.ravel()  # Assuming y_test is the first column after scaling
                        y_test = scaler.inverse_transform(placeholder)[:, 0]

                        placeholder[:, 0] = y_predicted.ravel()
                        y_predicted = scaler.inverse_transform(placeholder)[:, 0]

                        # Store the results
                        results.append({
                            'Model': model_name,
                            'Optimiser': 'None',

                            # Statistics
                            'Year': year,
                            'Month': month,
                            'Feature Set': feature_set,

                            # Monthly data - This is an hourly record of predicted data for the month from the day cutoff
                            'day_cutoff': day_cutoff,
                            'y_test': y_test,
                            'y_pred': y_predicted
                        })

                        print(f"Completed {feature_set} on model {model_name} for {calendar.month_name[month]} {year}")

        self._create_visualisations_from_results(results)

    def _create_visualisations_from_results(self, results) -> None:
        df_raw_results = pd.DataFrame(results)

        df_results = pd.DataFrame(columns=[
            'Model',
            'Optimiser',
            'Date & Time',
            'Feature Set',
            'Original',
            'Predicted'
        ])
        df_metrics = pd.DataFrame(columns=[
            'Model',
            'Optimiser',
            'Feature Set',

            'MSE',
            'MAE',
            'RMSE',
            'R2'
        ])

        for row_info in df_raw_results.iterrows():
            row = row_info[1]

            day = row.iloc[5] # day_cutoff
            hour = 0

            for y_test, y_pred in zip(row.iloc[6], row.iloc[7]):
                df_row_results = pd.DataFrame([{
                    'Model': row.iloc[0],
                    'Optimiser': row.iloc[1],

                    'Feature Set': row.iloc[4],
                    'Original': y_test,
                    'Predicted': y_pred,

                    'Year': row.iloc[2],
                    'Month': row.iloc[3],
                    'Day': day,
                    'Hour': hour
                }])

                df_row_results['Date & Time'] = pd.to_datetime(df_row_results[['Year', 'Month', 'Day', 'Hour']])
                df_row_results.drop(columns=['Year', 'Month', 'Day', 'Hour'], inplace=True)

                df_results = pd.concat([df_results, df_row_results], ignore_index=True)

                hour += 1

                if hour == 24:
                    day += 1
                    hour = 0

        # Metrics Calculation
        feature_sets = df_results['Feature Set'].unique()
        models = df_results['Model'].unique()
        optimisers = df_results['Optimiser'].unique()

        for feature_set in feature_sets:
            for model in models:
                for optimiser in optimisers:
                    df_relevant_records = df_results.loc[
                        (df_results['Model'] == model) & (df_results['Optimiser'] == optimiser) & (df_results['Feature Set'] == feature_set)
                    ]

                    y_test = df_relevant_records['Original'].values
                    y_predicted = df_relevant_records['Predicted'].values

                    try:
                        mse = mean_squared_error(y_test, y_predicted)
                    except ValueError:
                        mse = np.nan

                    try:
                        mae = mean_absolute_error(y_test, y_predicted)
                    except ValueError:
                        mae = np.nan

                    try:
                        rmse = np.sqrt(mse)
                    except ValueError:
                        rmse = np.nan

                    try:
                        r2 = r2_score(y_test, y_predicted)
                    except ValueError:
                        r2 = np.nan

                    df_row_results = pd.DataFrame([{
                        'Model': model,
                        'Optimiser': optimiser,

                        'Feature Set': feature_set,
                        'MSE': mse,
                        'MAE': mae,
                        'RMSE': rmse,
                        'R2': r2
                    }])

                    df_metrics = pd.concat([df_metrics, df_row_results], ignore_index=True)

        df_metrics.to_csv('metrics.csv')

        # Begin generating visualisations
        for feature_set in feature_sets:
            grid = sns.relplot(
                data=df_results.loc[df_results['Feature Set'] == feature_set], x='Original', y='Predicted',
                col='Model', hue='Optimiser', style='Optimiser',
                kind='scatter'
            )

            grid.figure.suptitle(f'Original vs Predicted - {feature_set}')

            plt.show()

    def _create_model_config(self) -> dict:
        model_config: dict = {
            ConfigOption.DISABLE_CACHE: self.config[ConfigOption.DISABLE_CACHE]
        }

        return model_config

    def _create_model(self, model_name: str) -> PredictionModel:
        match model_name:
            case PredictionModelName.DECISION_TREE:
                return RandomForestModel(self._create_model_config())
            case PredictionModelName.EXTRA_TREE:
                return ExtraTreeModel(self._create_model_config())
            case PredictionModelName.RANDOM_FOREST:
                return RandomForestModel(self._create_model_config())
            case PredictionModelName.LSTM:
                return LSTMModel(self._create_model_config())
            case PredictionModelName.BILSTM:
                return BiLSTMModel(self._create_model_config())

        raise Exception(f'No model for name: \'{model_name}\'')

    def _get_feature_set_columns(self, feature_set: str, dataset: DataFrame) -> list[str]:
        match feature_set:
            case FeatureSet.HISTORICAL_ONLY:
                return ['MW']
            case FeatureSet.HISTORICAL_WEATHER:
                return ['MW', 'Temperature', 'Humidity', 'Pressure', 'Wind Speed', 'Wind Direction', 'Downward Irradiance', 'Clearness Index', 'UV Index']
            case FeatureSet.ALL_FEATURES:
                return [col for col in dataset.columns if col not in ['DATE-TIME', 'Date & Time', 'DATE', 'Date']]
            case FeatureSet.HISTORICAL_TEMPERATURE:
                return ['MW', 'Temperature']
            case FeatureSet.HISTORICAL_WIND_SPEED:
                return ['MW', 'Wind Speed']
            case FeatureSet.HISTORICAL_HUMIDITY:
                return ['MW', 'Humidity']
            case FeatureSet.HISTORICAL_PRESSURE:
                return ['MW', 'Pressure']
            case FeatureSet.HISTORICAL_IRRADIANCE:
                return ['MW', 'Downward Irradiance']
            case FeatureSet.HISTORICAL_CLEARNESS:
                return ['MW', 'Clearness Index']
            case FeatureSet.HISTORICAL_UV_INDEX:
                return ['MW', 'UV Index']
            case FeatureSet.HISTORICAL_WITH_LAG_FEATURES:
                return ['MW', 'Lag_1H_MW', 'Lag_24H_MW', 'Lag_1H_Temperature', 'Lag_24H_Temperature']
            case FeatureSet.HISTORICAL_WITH_TIME_OF_DAY_AND_LAG_FEATURES:
                return ['MW', 'Hour', 'Lag_1H_MW', 'Lag_24H_MW']
            case _:
                raise Exception(f'Unknown feature set: \'{feature_set}\'')

if __name__ == '__main__':
    ModelRunner({
        ConfigOption.ENABLED_MODELS: ['Random Forest'],
        ConfigOption.DISABLE_CACHE: True,
        ConfigOption.PROBLEM: 'Supply',
        ConfigOption.FEATURE_SETS: ['Full Feature Set'],
    }).run_models()
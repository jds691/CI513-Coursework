from enum import StrEnum


class PredictionModelName(StrEnum):
    DECISION_TREE = 'Decision Tree'
    EXTRA_TREE = 'Extra Tree'
    RANDOM_FOREST = 'Random Forest'
    BILSTM = 'BiLSTM'
    LSTM = 'LSTM'

class ConfigOption(StrEnum):
    ENABLED_MODELS = 'Enabled Models'
    PROBLEM = 'Problem'
    DISABLE_CACHE = 'Disable Cache'
    FEATURE_SETS = 'Feature Sets'

class Problem(StrEnum):
    SUPPLY = 'Supply'
    DEMAND = 'Demand'

class FeatureSet(StrEnum):
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

class RunnerDisableCache(StrEnum):
    NO = 'No'
    YES = 'Yes'

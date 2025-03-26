from enum import StrEnum, auto

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class Dataset:
    def __init__(self, path: str):
        self.path = path
        self.is_cleaned = False

    def clean(self) -> None:
        pass

    def visualise(self, figures=str | list[str]) -> None:
        pass

class WeatherDataset(Dataset):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.dataset = pd.read_csv(self.path, skiprows=16)

    def clean(self) -> None:
        if self.is_cleaned:
            return

        self.dataset.rename(columns={
            'YEAR': 'Year',
            'MO': 'Month',
            'DY': 'Day',
            'HR': 'Hour',
            'ALLSKY_SFC_SW_DWN': 'Downward Irradiance',
            'ALLSKY_SFC_UV_INDEX': 'UV Index',
            'ALLSKY_KT': 'Clearness Index',
            'T2M': 'Temperature',
            'QV2M': 'Humidity',
            'PS': 'Pressure',
            'WS10M': 'Wind Speed',
            'WD10M': 'Wind Direction'
        }, inplace=True)
        self.dataset['DATE-TIME'] = pd.to_datetime(self.dataset[['Year', 'Month', 'Day', 'Hour']])
        self.dataset = self.dataset.drop(columns=['Year', 'Month', 'Day', 'Hour'])

        self.is_cleaned = True

    def visualise(self, figures=str | list[str]) -> None:
        print(self.dataset.head())


class EnergyDataset(Dataset):
    def __init__(self, path: str, datetime_column: str='DATE-TIME') -> None:
        super().__init__(path)
        self.dataset = pd.read_excel(path)
        self.dataset.rename(columns={datetime_column: 'DATE-TIME'}, inplace=True)

    def clean(self) -> None:
        if self.is_cleaned:
            return

        # Force MW to be a numeric value
        self.dataset['MW'] = pd.to_numeric(self.dataset['MW'], errors='coerce')

        # Convert DateTime values to Pandas DataTime values
        self.dataset['DATE-TIME'] = pd.to_datetime(self.dataset['DATE-TIME'])

        # Extract date to separate column
        self.dataset['Date'] = self.dataset['DATE-TIME'].dt.date

        # Calculate daily average and merge back into table
        daily_avg = self.dataset.groupby('Date')['MW'].mean().reset_index()
        self.dataset = pd.merge(self.dataset, daily_avg, on='Date', suffixes=('', '_Daily_AVG'))

        self.dataset['Missing_MW'] = self.dataset['MW'].isna() | (self.dataset['MW'] == 0)
        self.dataset['Weekday'] = self.dataset['DATE-TIME'].dt.dayofweek  # Monday=0, Sunday=6
        self.dataset['Hour'] = self.dataset['DATE-TIME'].dt.hour

        # Calculate the average MW for each hour of each day of the week
        hourly_avg = self.dataset.groupby(['Weekday', 'Hour'])['MW'].mean()

        # Function to fill missing values with the hourly average for that specific day of the week and hour
        def fill_missing(row):
            if row['Missing_MW']:
                # Get the corresponding average for that day of the week and hour
                return hourly_avg.loc[row['Weekday'], row['Hour']]
            else:
                return row['MW']

        # Apply the function to fill missing values
        self.dataset['MW'] = self.dataset.apply(fill_missing, axis=1)

        # Drop the auxiliary columns used for calculation
        self.dataset = self.dataset.drop(columns=['Missing_MW', 'Weekday', 'Hour'])

        self.is_cleaned = True

    def visualise(self, figures=str | list[str]) -> None:
        if isinstance(figures, str):
            if figures == 'all':
                figures = list(map(lambda enum: enum.value, self.Visualisations))
            else:
                figures = [figures]

        for figure in figures:
            match figure:
                case self.Visualisations.HEAD:
                    print(self.dataset.head())
                case self.Visualisations.AVG_LOAD:
                    # Plot daily average load
                    daily_avg_plot = sns.lineplot(x='Date', y='MW', data=self.dataset)
                    daily_avg_plot.set(xlabel='Date', ylabel='Average MW')
                    daily_avg_plot.set_title('Average Daily PV load in 2021')

                    plt.show()
                case self.Visualisations.LOW_POWER:
                    # Filter days when the average PV supply is less than 50 MW
                    low_supply_days = self.dataset[self.dataset['MW_Daily_AVG'] < 50]

                    low_power_plot = sns.barplot(data=low_supply_days, x='Date', y='MW_Daily_AVG')
                    low_power_plot.set(xlabel='Date', ylabel='Average MW')
                    low_power_plot.set_title('Low Power PV Load in 2021 (< 50 MW)')

                    plt.show()
                case _:
                    print(f'Unknown visualisation: \'{figure}\'')

    class Visualisations(StrEnum):
        HEAD = auto()
        AVG_LOAD = auto()
        LOW_POWER = auto()

class MergedDataset(Dataset):
    def __init__(self, energy_dataset: EnergyDataset, weather_dataset: WeatherDataset) -> None:
        super().__init__('')

        energy_dataset.clean()
        weather_dataset.clean()

        self.dataset = pd.merge(energy_dataset.dataset, weather_dataset.dataset, on='DATE-TIME', how='left')

    def clean(self) -> None:
        if self.is_cleaned:
            return

        # Extract time-based features
        self.dataset['Month'] = self.dataset['DATE-TIME'].dt.month
        self.dataset['Day'] = self.dataset['DATE-TIME'].dt.day
        self.dataset['Hour'] = self.dataset['DATE-TIME'].dt.hour
        self.dataset['Day Of Week'] = self.dataset['DATE-TIME'].dt.dayofweek
        self.dataset['Is Weekend'] = self.dataset['Day Of Week'].apply(lambda x: 1 if x >= 5 else 0)

        # Interaction features
        self.dataset['Irradiance_Temperature'] = self.dataset['Downward Irradiance'] * self.dataset['Temperature']
        self.dataset['Humidity_Wind Speed'] = self.dataset['Humidity'] * self.dataset['Wind Speed']

        # Feature: Temperature difference over hours
        self.dataset['Temperature Difference'] = self.dataset['Temperature'].diff().fillna(0)

        # Feature: Wind speed and direction combined
        self.dataset['Wind'] = self.dataset['Wind Speed'] * (self.dataset['Wind Direction'].apply(np.radians)).apply(np.cos)

        # Replace -999 values in the UV Index column with 0
        self.dataset['UV Index'] = self.dataset['UV Index'].replace(-999, 0)

        # 1. Time-Based Features
        # Day of the Year
        self.dataset['Day Of Year'] = self.dataset['DATE-TIME'].dt.dayofyear

        # holidays = []
        # self.dataset['Is_Holiday'] = self.dataset['DATE'].isin(holidays).astype(int)

        # 2. Weather-Based Interaction Features
        self.dataset['Humidity_Temperature'] = self.dataset['Humidity'] * self.dataset['Temperature']
        self.dataset['WindSpeed_Temperature'] = self.dataset['Wind Speed'] * self.dataset['Temperature']
        self.dataset['Irradiance_Cloud Cover'] = self.dataset['Downward Irradiance'] * (
                    1 - self.dataset['Clearness Index'])  # Assuming 1 - Clearness Index represents cloud cover

        # 3. Lag Features
        # Lagged PV Supply (e.g., 1 hour and 24 hours)
        self.dataset['Lag_1H_MW'] = self.dataset['MW'].shift(1)
        self.dataset['Lag_24H_MW'] = self.dataset['MW'].shift(24)

        # Lagged Weather Conditions (e.g., temperature)
        self.dataset['Lag_1H_Temperature'] = self.dataset['Temperature'].shift(1)
        self.dataset['Lag_24H_Temperature'] = self.dataset['Temperature'].shift(24)

        # 4. Rolling Statistics
        # Rolling Mean of PV Supply over 24 hours and 7 days
        self.dataset['Rolling_24H_MW_Mean'] = self.dataset['MW'].rolling(window=24).mean()
        self.dataset['Rolling_7D_MW_Mean'] = self.dataset['MW'].rolling(window=24*7).mean()

        # Rolling Standard Deviation of PV Supply over 24 hours
        self.dataset['Rolling_24H_MW_Std'] = self.dataset['MW'].rolling(window=24).std()

        # 5. Cyclic Features
        # Sine/Cosine Transformations of Hour and DayOfYear
        self.dataset['Hour_Sin'] = np.sin(2 * np.pi * self.dataset['Hour'] / 24)
        self.dataset['Hour_Cos'] = np.cos(2 * np.pi * self.dataset['Hour'] / 24)
        self.dataset['Day Of Year_Sin'] = np.sin(2 * np.pi * self.dataset['Day Of Year'] / 365)
        self.dataset['Day Of Year_Cos'] = np.cos(2 * np.pi * self.dataset['Day Of Year'] / 365)

        # 6. Trend Features
        # Cumulative Sum of PV Supply
        self.dataset['Cumulative_MW'] = self.dataset['MW'].cumsum()

        # Difference from Moving Average (e.g., 24-hour moving average)
        self.dataset['Diff_24H_MW_MA'] = self.dataset['MW'] - self.dataset['Rolling_24H_MW_Mean']

        # Fill any resulting NaN values (especially for lagged and rolling features)
        self.dataset.fillna(0, inplace=True)

        self.is_cleaned = True

    def visualise(self, figures=str | list[str]) -> None:
        if isinstance(figures, str):
            if figures == 'all':
                figures = list(map(lambda enum: enum.value, self.Visualisations))
            else:
                figures = [figures]

        numeric_cols = self.dataset.select_dtypes(include=[np.number])

        for figure in figures:
            match figure:
                case self.Visualisations.HEAD:
                    # Display the first few rows to verify the new features
                    print("Dataset with new features:")
                    print(self.dataset.head())
                case self.Visualisations.NUMERIC_CORRELATION:
                    correlation = numeric_cols.corr()
                    sns.heatmap(correlation, annot=True, fmt=".2f")
                    plt.show()
                case self.Visualisations.FEATURE_DISTRIBUTION:
                    # Plotting box plots
                    sns.boxplot(numeric_cols)
                    plt.title('Boxplot for Numeric Features to Detect Outliers')
                    plt.xticks(rotation=90)
                    plt.show()

                    # Plot histograms for all numeric features
                    numeric_cols.hist()
                    plt.suptitle('Histograms of Numeric Features')
                    plt.show()
                case self.Visualisations.TARGET_DISTRIBUTION:
                    sns.histplot(self.dataset['MW'], kde=True)
                    plt.title('Distribution of Target Variable (MW)')
                    plt.show()
                case _:
                    print(f'Unknown visualisation: \'{figure}\'')

    class Visualisations(StrEnum):
        HEAD = auto()
        NUMERIC_CORRELATION = auto()
        FEATURE_DISTRIBUTION = auto()
        TARGET_DISTRIBUTION = auto()


if __name__ == '__main__':
    energy_demand: EnergyDataset = EnergyDataset('data/Sakakah 2021 Demand dataset.xlsx')
    energy_demand.clean()
    energy_demand.visualise([
        EnergyDataset.Visualisations.HEAD,
        EnergyDataset.Visualisations.AVG_LOAD
    ])

    energy_supply: EnergyDataset = EnergyDataset('data/Sakakah 2021 PV Supply dataset.xlsx', datetime_column='Date & Time')
    energy_supply.clean()
    energy_supply.visualise([
        EnergyDataset.Visualisations.HEAD,
        EnergyDataset.Visualisations.LOW_POWER
    ])

    weather_demand: WeatherDataset = WeatherDataset('data/Sakakah 2021 weather dataset Demand.csv')
    weather_demand.clean()
    weather_demand.visualise('all')

    weather_supply: WeatherDataset = WeatherDataset('data/Sakakah 2021 weather dataset.csv')
    weather_supply.clean()
    weather_supply.visualise('all')

    merged_supply: MergedDataset = MergedDataset(energy_supply, weather_supply)
    merged_supply.clean()
    merged_supply.visualise('all')

    merged_demand: MergedDataset = MergedDataset(energy_demand, weather_demand)
    merged_demand.clean()
    merged_demand.visualise('all')
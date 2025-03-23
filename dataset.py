from enum import StrEnum, auto

import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns


class Dataset:
    def __init__(self, path: str):
        self.path = path

    def clean(self) -> None:
        pass

    def visualise(self, figures=str | list[str]) -> None:
        pass

class WeatherDataset(Dataset):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.dataset = pd.read_csv(self.path, skiprows=16)

    def clean(self) -> None:
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

    def visualise(self, figures=str | list[str]) -> None:
        print(self.dataset.head())


class EnergyDataset(Dataset):
    def __init__(self, path: str, datetime_column: str='DATE-TIME') -> None:
        super().__init__(path)
        self.dataset = pd.read_excel(path)
        self.dataset.rename(columns={datetime_column: 'DATE-TIME'}, inplace=True)

    def clean(self) -> None:
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

    def visualise(self, figures=str | list[str]) -> None:
        if isinstance(figures, str):
            if figures == 'all':
                # TODO
                pass
            else:
                figures = [figures]

        for figure in figures:
            match figure:
                case EnergyDataset.Visualisations.HEAD:
                    print(self.dataset.head())
                case EnergyDataset.Visualisations.AVG_LOAD:
                    # Plot daily average load
                    daily_avg_plot = sns.lineplot(x='Date', y='MW', data=self.dataset)
                    daily_avg_plot.set(xlabel='Date', ylabel='Average MW')
                    daily_avg_plot.set_title('Average Daily PV load in 2021')

                    plt.show()
                case EnergyDataset.Visualisations.LOW_POWER:
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

    weather_supply: WeatherDataset = WeatherDataset('data/Sakakah 2021 weather dataset.csv')
    weather_supply.clean()
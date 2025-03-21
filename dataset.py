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
        # Convert DateTime values to Pandas DataTime values
        self.dataset['DATE-TIME'] = pd.to_datetime(self.dataset['DATE-TIME'])

        # Extract date to separate column
        self.dataset['Date'] = self.dataset['DATE-TIME'].dt.date

        # Calculate daily average and merge back into table
        daily_avg = self.dataset.groupby('Date')['MW'].mean().reset_index()
        self.dataset = pd.merge(self.dataset, daily_avg, on='Date', suffixes=('', '_Daily_AVG'))

    def visualise(self, figures=str | list[str]) -> None:
        print(self.dataset.head())

        # Plot the daily average
        daily_avg_plot = sns.lineplot(x='Date', y='MW', data=self.dataset)
        daily_avg_plot.set(xlabel='Date', ylabel='Average MW')
        daily_avg_plot.set_title('Average Daily PV load in 2021')

        plt.show()

if __name__ == '__main__':
    energy_demand: EnergyDataset = EnergyDataset('data/Sakakah 2021 Demand dataset.xlsx')
    energy_demand.clean()

    energy_supply: EnergyDataset = EnergyDataset('data/Sakakah 2021 PV Supply dataset.xlsx', datetime_column='Date & Time')
    energy_supply.clean()

    weather_demand: WeatherDataset = WeatherDataset('data/Sakakah 2021 weather dataset Demand.csv')
    weather_demand.clean()

    weather_supply: WeatherDataset = WeatherDataset('data/Sakakah 2021 weather dataset.csv')
    weather_supply.clean()
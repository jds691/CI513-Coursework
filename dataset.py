import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns


class Dataset:
    def __init__(self, path: str):
        self.path = path

    def clean(self) -> DataFrame:
        pass

    def visualise(self, figures=str | list[str]) -> None:
        pass

class WeatherDataset(Dataset):
    pass

class EnergyDataset(Dataset):
    def __init__(self, path: str, datetime_column: str='DATE-TIME') -> None:
        super().__init__(path)
        self.dataset = pd.read_excel(path)

        self.datetime_column = datetime_column

    def clean(self) -> None:
        # Convert DateTime values to Pandas DataTime values
        self.dataset[self.datetime_column] = pd.to_datetime(self.dataset[self.datetime_column])

        # Extract date to separate column
        self.dataset['Date'] = self.dataset[self.datetime_column].dt.date

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
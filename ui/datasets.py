from typing import Callable

import survey

from dataset import EnergyDataset, WeatherDataset

def _visualise_energy_demand() -> None:
    dataset: EnergyDataset = EnergyDataset('data/Sakakah 2021 Demand dataset.xlsx')
    dataset.clean()
    dataset.visualise([
        EnergyDataset.Visualisations.HEAD,
        EnergyDataset.Visualisations.AVG_LOAD
    ])

def _visualise_weather_demand() -> None:
    dataset: WeatherDataset = WeatherDataset('data/Sakakah 2021 weather dataset Demand.csv')
    dataset.clean()
    dataset.visualise('all')

def _visualise_energy_supply() -> None:
    dataset: EnergyDataset = EnergyDataset('data/Sakakah 2021 PV Supply dataset.xlsx', datetime_column='Date & Time')
    dataset.clean()
    dataset.visualise([
        EnergyDataset.Visualisations.HEAD,
        EnergyDataset.Visualisations.LOW_POWER
    ])

def _visualise_weather_supply() -> None:
    dataset: WeatherDataset = WeatherDataset('data/Sakakah 2021 weather dataset.xlsx')
    dataset.clean()
    dataset.visualise('all')

def display_menu(on_exit: Callable[[], None]=lambda: ()) -> None:
    menu_options = (
        'Energy - Demand',
        'Weather - Demand',
        'Energy - Supply',
        'Weather - Supply',
        'Back'
    )

    menu_index = survey.routines.select('Visualise Dataset:', options=menu_options)

    match menu_index:
        case 0:
            _visualise_energy_demand()
            display_menu(on_exit=on_exit)
        case 1:
            _visualise_weather_demand()
            display_menu(on_exit=on_exit)
        case 2:
            _visualise_energy_supply()
            display_menu(on_exit=on_exit)
        case 1:
            _visualise_weather_supply()
            display_menu(on_exit=on_exit)
        case _:
            on_exit()

if __name__ == '__main__':
    display_menu()
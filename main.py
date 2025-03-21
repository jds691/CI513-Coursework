import survey

import ui.prediction
import ui.optimisation
import ui.datasets

from utils import clear_console

def _display_main_menu() -> None:
    menu_options = (
        'Prediction',
        'Optimisation',
        'Visualise Datasets',
        'Exit'
    )

    menu_index = survey.routines.select('Main Menu:', options=menu_options)

    match menu_index:
        case 0:
            ui.prediction.display_menu(on_exit=_display_main_menu)
        case 1:
            ui.optimisation.display_menu(on_exit=_display_main_menu)
        case 2:
            ui.datasets.display_menu(on_exit=_display_main_menu)
        case _:
            # Allows for exiting without error
            pass

if __name__ == '__main__':
    clear_console()

    _display_main_menu()
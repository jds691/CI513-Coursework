import time
from typing import Callable

import survey


def display_menu(on_exit: Callable[[], None]=None) -> None:
    menu_options = (
        'Temp',
        'Exit'
    )

    menu_index = survey.routines.select('Optimisation:', options=menu_options)

    match menu_index:
        case _:
            on_exit()
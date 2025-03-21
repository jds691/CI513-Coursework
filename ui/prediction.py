from typing import Callable

import survey


def display_menu(on_exit: Callable[[], None]=lambda: ()) -> None:
    menu_options = (
        'Temp',
        'Back'
    )

    menu_index = survey.routines.select('Prediction:', options=menu_options)

    match menu_index:
        case _:
            on_exit()


if __name__ == '__main__':
    display_menu()
from typing import Callable

import survey


def display_menu(on_exit: Callable[[], None]=lambda: ()) -> None:
    available_prediction_models = [
        'Decision Tree Regression'
    ]
    model_selection_widget = survey.widgets.Basket(options=available_prediction_models)

    available_problems = [
        'Supply',
        'Demand'
    ]
    problem_selection_widget = survey.widgets.Select(options=available_problems)

    disable_cache_options = [
        'No',
        'Yes'
    ]
    disable_cache_selection_widget = survey.widgets.Select(options=disable_cache_options)

    form_options = {
        'Enabled Models': model_selection_widget,
        'Problem': problem_selection_widget,
        'Disable Cache': disable_cache_selection_widget
    }

    prediction_config_data: dict = survey.routines.form('Prediction:', form=form_options)

    if survey.routines.inquire('Run prediction with the above settings?', default=True):
        # TODO: Run with config settings
        pass
    elif survey.routines.inquire('Return to main menu?', default=True):
        on_exit()
    else:
        display_menu(on_exit=on_exit)



if __name__ == '__main__':
    display_menu()
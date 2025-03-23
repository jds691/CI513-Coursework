from typing import Callable

import survey

from models.prediction import PredictionModelRunner, PredictionConfigOption, PredictionProblem, \
    PredictionModelDisableCache, PredictionModelName

def display_menu(on_exit: Callable[[], None]=lambda: ()) -> None:
    available_prediction_models = []
    for name in PredictionModelName:
        available_prediction_models.append(name.value)

    model_selection_widget = survey.widgets.Basket(options=available_prediction_models)

    available_problems = []
    for name in PredictionProblem:
        available_problems.append(name.value)

    problem_selection_widget = survey.widgets.Select(options=available_problems)

    disable_cache_options = []
    for option in PredictionModelDisableCache:
        disable_cache_options.append(option.value)

    disable_cache_selection_widget = survey.widgets.Select(options=disable_cache_options)

    form_options = {
        PredictionConfigOption.ENABLED_MODELS : model_selection_widget,
        PredictionConfigOption.PROBLEM: problem_selection_widget,
        PredictionConfigOption.DISABLE_CACHE: disable_cache_selection_widget
    }

    prediction_config_data: dict = survey.routines.form('Prediction:', form=form_options)

    if survey.routines.inquire('Run prediction with the above settings?', default=True):
        PredictionModelRunner(prediction_config_data).run_models()
        on_exit()
    elif survey.routines.inquire('Return to main menu?', default=True):
        on_exit()
    else:
        display_menu(on_exit=on_exit)



if __name__ == '__main__':
    display_menu()
from typing import Callable

import survey

from models.prediction import PredictionModelRunner, PredictionConfigOption, PredictionProblem, \
    PredictionModelDisableCache, PredictionModelName, PredictionFeatureSets


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

    available_feature_sets = []
    for feature_set in PredictionFeatureSets:
        available_feature_sets.append(feature_set.value)

    feature_sets_selection_widget = survey.widgets.Basket(options=available_feature_sets)

    form_options = {
        PredictionConfigOption.ENABLED_MODELS : model_selection_widget,
        PredictionConfigOption.PROBLEM: problem_selection_widget,
        PredictionConfigOption.DISABLE_CACHE: disable_cache_selection_widget,
        PredictionConfigOption.FEATURE_SETS: feature_sets_selection_widget
    }

    prediction_config_data: dict = survey.routines.form('Prediction:', form=form_options)

    model_names = map(lambda model_index: available_prediction_models[model_index], prediction_config_data[PredictionConfigOption.ENABLED_MODELS])
    prediction_config_data[PredictionConfigOption.ENABLED_MODELS] = list(model_names)

    prediction_config_data[PredictionConfigOption.PROBLEM] = available_problems[prediction_config_data[PredictionConfigOption.PROBLEM]]
    prediction_config_data[PredictionConfigOption.DISABLE_CACHE] = disable_cache_options[prediction_config_data[PredictionConfigOption.DISABLE_CACHE]]

    feature_sets = map(lambda set_index: available_feature_sets[set_index],
                      prediction_config_data[PredictionConfigOption.FEATURE_SETS])
    prediction_config_data[PredictionConfigOption.FEATURE_SETS] = list(feature_sets)

    if survey.routines.inquire('Run prediction with the above settings?', default=True):
        PredictionModelRunner(prediction_config_data).run_models()
        on_exit()
    elif survey.routines.inquire('Return to main menu?', default=True):
        on_exit()
    else:
        display_menu(on_exit=on_exit)



if __name__ == '__main__':
    display_menu()
from typing import Callable

import survey

from models.config import ConfigOption, Problem, \
    RunnerDisableCache, PredictionModelName, FeatureSet
from models.runner import ModelRunner

def display_menu(on_exit: Callable[[], None]=lambda: ()) -> None:
    available_prediction_models = []
    for name in PredictionModelName:
        available_prediction_models.append(name.value)

    model_selection_widget = survey.widgets.Basket(options=available_prediction_models)

    available_problems = []
    for name in Problem:
        available_problems.append(name.value)

    problem_selection_widget = survey.widgets.Select(options=available_problems)

    disable_cache_options = []
    for option in RunnerDisableCache:
        disable_cache_options.append(option.value)

    disable_cache_selection_widget = survey.widgets.Select(options=disable_cache_options)

    available_feature_sets = []
    for feature_set in FeatureSet:
        available_feature_sets.append(feature_set.value)

    feature_sets_selection_widget = survey.widgets.Basket(options=available_feature_sets)

    form_options = {
        ConfigOption.ENABLED_MODELS : model_selection_widget,
        ConfigOption.PROBLEM: problem_selection_widget,
        ConfigOption.DISABLE_CACHE: disable_cache_selection_widget,
        ConfigOption.FEATURE_SETS: feature_sets_selection_widget
    }

    runner_config_data: dict = survey.routines.form('Run Models:', form=form_options)

    model_names = map(lambda model_index: available_prediction_models[model_index], runner_config_data[ConfigOption.ENABLED_MODELS])
    runner_config_data[ConfigOption.ENABLED_MODELS] = list(model_names)

    runner_config_data[ConfigOption.PROBLEM] = available_problems[runner_config_data[ConfigOption.PROBLEM]]
    runner_config_data[ConfigOption.DISABLE_CACHE] = disable_cache_options[runner_config_data[ConfigOption.DISABLE_CACHE]]

    feature_sets = map(lambda set_index: available_feature_sets[set_index],
                       runner_config_data[ConfigOption.FEATURE_SETS])
    runner_config_data[ConfigOption.FEATURE_SETS] = list(feature_sets)

    if survey.routines.inquire('Run models with the above settings?', default=True):
        ModelRunner(runner_config_data).run_models()
        on_exit()
    elif survey.routines.inquire('Return to main menu?', default=True):
        on_exit()
    else:
        display_menu(on_exit=on_exit)



if __name__ == '__main__':
    display_menu()
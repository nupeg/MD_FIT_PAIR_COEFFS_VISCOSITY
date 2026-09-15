from typing import final

from viscofit.utils import ask_user_choice
from viscofit.settings import DEFAULT_SETTINGS_FILEPATH, Settings

from viscofit.experiments import (
    Experiment, 
    ExperimentRegister,
    fit_params,
    preliminar_tests,
    evaluate_madrid_2019
)


@final
class AvailableExperiments(ExperimentRegister):
    E1 = Experiment(
        func=preliminar_tests, 
        name='preliminar-tests', 
        description='Fit Lennard-Jones parameters with Bayesian Optimization using preliminar-tests data'
    )
    E2 = Experiment(
        func=fit_params, 
        name='fit-params', 
        description='Fit Lennard-Jones parameters with Bayesian Optimization using fit-params data'
    )
    E3 = Experiment(
        func=evaluate_madrid_2019, 
        name='evaluate-madrid-2019', 
        description='Calculate viscosity using madrid-2019 coefficents'
    )


def main() -> None:
    options = AvailableExperiments.catalog()
    experiment_name = ask_user_choice(options)

    config_handler = AvailableExperiments.take(experiment_name)
    config = Settings.from_file(DEFAULT_SETTINGS_FILEPATH)

    config_handler(config)

if __name__ == '__main__':
    main()


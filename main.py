from typing import final

from viscofit.utils import ask_user_choice, docstring
from viscofit.settings import DEFAULT_SETTINGS_FILEPATH, Settings

from viscofit.experiments import (
    pure_water_simulations
)
from viscofit.experiments_helpers import ExperimentRegister, Experiment


@final
class AvailableExperiments(ExperimentRegister):
    experiments = [
        Experiment(
            func=pure_water_simulations, 
            name='pure-water-simulations', 
            description=docstring(pure_water_simulations)
        )
    ]
    

def main() -> None:
    options = AvailableExperiments.catalog()
    experiment_name = ask_user_choice(options)

    config_handler = AvailableExperiments.take(experiment_name)
    config = Settings.from_file(DEFAULT_SETTINGS_FILEPATH)

    config_handler(config)

if __name__ == '__main__':
    main()


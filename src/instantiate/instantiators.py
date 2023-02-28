from abc import ABCMeta, abstractmethod
from itertools import chain
from typing import Any, Dict, List, Tuple, Union, TypeVar, Sequence
from optuna import Trial
# from lightgbm import LGBMModel, LGBMClassifier, LGBMRegressor, LGBMRanker
# from xgboost import XGBModel, XGBClassifier, XGBRegressor, XGBRanker

Parameter = Union[int, float, bool, str, Sequence[Any]]
ParameterSpace = Dict[str, Parameter]
T = TypeVar("T")


class AbstractInstanceCreator(ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def _get_parameter_values(self) -> Tuple[str, Parameter]:
        pass

    @abstractmethod
    def __call__(self, trial: Trial) -> Any:
        pass


class InstanceCreator(AbstractInstanceCreator):
    def __init__(self, space: Dict[str, ParameterSpace], object_type: type) -> None:
        self.space = space
        self.object_type = object_type
        return

    def _get_parameter_values(
        self, trial: Trial, parameter_name: str, parameter_space: ParameterSpace
    ) -> Tuple[str, Parameter]:
        if "distribution" in parameter_space.keys():
            is_categorical = False
        elif len(parameter_space.keys()) == 1:
            is_categorical = True
        else:
            raise ValueError(
                f'The space passed for the parameter "{parameter_name}" is not valid.'
            )

        if is_categorical:
            return parameter_name, trial.suggest_categorical(
                parameter_name, [*chain(*parameter_space.values())]
            )
        else:
            if parameter_space["distribution"] == "int":
                return parameter_name, trial.suggest_int(
                    parameter_name,
                    parameter_space["min"],
                    parameter_space["max"],
                    step=parameter_space.get("step", 1),
                    log=parameter_space.get("log", False),
                )
            elif parameter_space["distribution"] == "float":
                return parameter_name, trial.suggest_float(
                    parameter_name,
                    parameter_space["min"],
                    parameter_space["max"],
                    step=parameter_space.get("step", 1),
                    log=parameter_space.get("log", False),
                )

    def __call__(self, trial: Trial) -> T:
        params = {
            name: value
            for name, value in [
                self._get_parameter_values(trial, parameter_name, parameter_space)
                for parameter_name, parameter_space in self.space.items()
            ]
        }

        return self.object_type(**params)


# def instantiate_lgbm(trial: Trial, model: str, random_state: int = 42) -> LGBMModel:
#     params = {
#         "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "rf"]),
#         "n_estimators": trial.suggest_int("n_estimators", 50, 1500),
#         "learning_rate": trial.suggest_float("learning_rate", 1e-5, 0.01),
#         "num_leaves": trial.suggest_int("num_leaves", 7, 31, step=2),
#         "max_depth": trial.suggest_int("max_depth", 1, 7),
#         "min_child_samples": trial.suggest_int(
#             "min_child_samples", 5, 10
#         ),  # The limits should loosely depend on the dataset size
#         "reg_alpha": trial.suggest_int("reg_alpha", 0, 100),
#         "reg_lambda": trial.suggest_int("reg_lambda", 0, 100),
#         "min_split_gain": trial.suggest_float("min_split_gain", 0, 15),
#         "subsample": trial.suggest_float("subsample", 0.4, 1, step=0.05),
#         "subsample_freq": trial.suggest_int("subsample_freq", 1, 100),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1, step=0.05),
#         "colsample_bynode": trial.suggest_float("colsample_bynode", 0.2, 1, step=0.05),
#         "linear_tree": trial.suggest_categorical("linear_tree", [True, False]),
#         "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
#         "random_state": random_state,
#     }

#     if model == "classifier":
#         return LGBMClassifier(**params)
#     elif model == "regressor":
#         return LGBMRegressor(**params)
#     elif model == "ranker":
#         return LGBMRanker(**params)

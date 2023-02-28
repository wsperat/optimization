from unittest import TestCase
from unittest.mock import Mock, MagicMock
from typing import Dict

from optuna import Trial
from src.instantiate.instantiators import (
    AbstractInstanceCreator,
    InstanceCreator,
    Parameter,
    ParameterSpace,
)


class TestAbstractInstanceCreator(TestCase):
    def test__get_parameter_values(self):

        creator = AbstractInstanceCreator()
        with self.assertRaises(TypeError):
            creator._get_parameter_values()

    def test__call(self):

        creator = AbstractInstanceCreator()
        with self.assertRaises(TypeError):
            creator.__call__()


class TestInstanceCreator(TestCase):
    def setUp(self):
        self.space = {
            "a": {"distribution": "int", "min": 1, "max": 10},
            "b": {"distribution": "float", "min": 0.0, "max": 1.0},
            "c": {"foo", "bar", "baz"}
        }
        self.obj_type = MagicMock()
        self.instance_creator = InstanceCreator(self.space, self.obj_type)

    def test_get_parameter_values_categorical(self):
        # Test categorical parameter value retrieval.
        trial = MagicMock()
        trial.suggest_categorical.return_value = "foo"
        parameter_name = "c"
        parameter_space = self.space[parameter_name]
        result = self.instance_creator._get_parameter_values(
            trial, parameter_name, parameter_space
        )
        trial.suggest_categorical.assert_called_once_with(
            parameter_name, {"foo", "bar", "baz"}
        )
        self.assertEqual(result, ("a", "foo"))

    def test_get_parameter_values_int(self):
        # Test integer parameter value retrieval.
        trial = MagicMock()
        trial.suggest_int.return_value = 5
        parameter_name = "a"
        parameter_space = self.space[parameter_name]
        result = self.instance_creator._get_parameter_values(
            trial, parameter_name, parameter_space
        )
        trial.suggest_int.assert_called_once_with(
            parameter_name, 1, 10, step=1, log=False
        )
        self.assertEqual(result, ("a", 5))

    def test_get_parameter_values_float(self):
        # Test float parameter value retrieval.
        trial = MagicMock()
        trial.suggest_float.return_value = 0.5
        parameter_name = "b"
        parameter_space = self.space[parameter_name]
        result = self.instance_creator._get_parameter_values(
            trial, parameter_name, parameter_space
        )
        trial.suggest_float.assert_called_once_with(
            parameter_name, 0.0, 1.0, step=1, log=False
        )
        self.assertEqual(result, ("b", 0.5))

    def test_get_parameter_values_invalid(self):
        # Test that an error is raised when an invalid parameter space is passed.
        trial = MagicMock()
        parameter_name = "c"
        parameter_space = {"distribution": "invalid"}
        with self.assertRaises(ValueError):
            self.instance_creator._get_parameter_values(
                trial, parameter_name, parameter_space
            )

    def test_call(self):
        # Test the __call__ method.
        trial = MagicMock()
        trial.suggest_int.side_effect = [2, 5]
        trial.suggest_float.side_effect = [0.5, 0.75]
        self.obj_type.return_value = MagicMock()
        result = self.instance_creator(trial)
        self.obj_type.assert_called_once_with(a=2, b=0.5)
        self.assertIsInstance(result, MagicMock)

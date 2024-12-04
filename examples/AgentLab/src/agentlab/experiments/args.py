import copy
from abc import ABC
from dataclasses import fields, is_dataclass
from itertools import product
from typing import Any

import numpy as np


class CrossProd:
    """Use to specify that this will be part of a cross product"""

    def __init__(self, elements):
        self.elements = elements


class Distribution(ABC):
    """Generic Class to identify that this is a distribution"""

    def sample(self):
        pass


class Choice(Distribution):
    """Use to specify that this will be sampled from a list of elements"""

    def __init__(self, elements, p=None):
        self.elements = elements
        self.p = p

    def sample(self, rng=np.random):
        return rng.choice(self.elements, p=self.p)


def _find_cprod_with_paths(obj, path=None):
    """Find all the CrossProd objects and their paths in the given object.

    Args:
        obj (Any): The object to search for CrossProd objects.
        path (List[str]): The path to the current object.

    Returns:
        List[Tuple[List[str], CrossProd]]: A list of tuples where the first element is the path to the CrossProd
            object and the second element is the CrossProd object.
    """
    if path is None:
        path = []

    if isinstance(obj, CrossProd):
        return [(path, obj)]

    cprod_paths = []
    if is_dataclass(obj):
        for field in fields(obj):
            field_value = getattr(obj, field.name)
            cprod_paths += _find_cprod_with_paths(field_value, path + [field.name])
    elif isinstance(obj, dict):
        for key, value in obj.items():
            cprod_paths += _find_cprod_with_paths(value, path + [key])

    return cprod_paths


def _set_value(obj, path, value):
    """Set the value of the given path in the given object to the given value."""
    for key in path[:-1]:
        if isinstance(obj, dict):
            obj = obj[key]
        else:
            obj = getattr(obj, key)
    if isinstance(obj, dict):
        obj[path[-1]] = value
    else:
        setattr(obj, path[-1], value)


def expand_cross_product(obj: Any | list[Any]):
    """Expand the given object into a list of objects with all combinations of
    CrossProd objects.

    This function will recursively search for CrossProd objects in the given
    object and create a list of objects with all combinations of CrossProd. It
    searches through dataclasses and dictionaries.

    Args:
        obj: Any | List[Any],
            The object to expand.

    Returns:
        List[Any]:
        A list of objects with all combinations of CrossProd objects.

    """

    if isinstance(obj, CrossProd):
        return obj.elements

    if isinstance(obj, list):
        obj_list = obj
    else:
        obj_list = [obj]

    result = []

    for obj in obj_list:
        cprod_paths = _find_cprod_with_paths(obj)
        if not cprod_paths:
            result.append(copy.deepcopy(obj))
            continue

        paths, cprod_objects = zip(*cprod_paths)
        combinations = product(*[cprod_obj.elements for cprod_obj in cprod_objects])

        # create a base object with empty fields to make fast deep copies from
        base_obj = copy.deepcopy(obj)
        for path in paths:
            _set_value(base_obj, path, None)

        for combo in combinations:
            new_obj = copy.deepcopy(base_obj)
            for path, value in zip(paths, combo):
                _set_value(new_obj, path, value)
            result.append(new_obj)

    return result


def sample_and_expand_cross_product(obj: Any | list[Any], n_samples: int):
    """This will sample first and then expand the cross product."""
    return expand_cross_product(sample_args(obj, n_samples))


def sample_args(obj: Any | list[Any], n_samples: int):
    """Sample the given object n_samples times. Each sample is a deep copy of
    the original object with any object of type Distribution replaced by a
    sample from that distribution.

    Args:
        obj: Any | List[Any],
            the object to sample
        n_samples: int,
            the number of samples to generate

    Returns:
        List[Any]:
        A list of n_samples objects with the given object.
    """

    if isinstance(obj, list):
        obj_list = obj
    else:
        obj_list = [obj]

    result = []

    for obj in obj_list:
        for _ in range(n_samples):
            result.append(_sample_single(copy.deepcopy(obj)))

    return result


def _sample_single(obj):
    """Sample the given object once."""
    if isinstance(obj, Distribution):
        return obj.sample()

    if is_dataclass(obj):
        for field in fields(obj):
            value = getattr(obj, field.name)
            sampled_value = _sample_single(value)
            setattr(obj, field.name, sampled_value)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = _sample_single(value)

    return obj


class Toggle:
    pass


TOGGLE = Toggle()


def _change_value(obj, path, value):
    """Set the value to the given path in the nested objects.

    Note: This doesn't work with list or dict or tuples. Only works with objects.

    Args:
        obj: the object to change the value
        path: the path to the value to change
        value: the value to change to

    Raises:
        ValueError: if the field is not found in the object
    """
    key_list = path.split(".")
    for key in key_list[:-1]:
        if key == "":
            continue
        obj = getattr(obj, key)
    # if dataclass, then set the value only if the field exists

    def _set(obj, key, value):
        if isinstance(value, Toggle):
            previous_value = getattr(obj, key)
            if not isinstance(previous_value, bool):
                raise ValueError(f"Toggle object {obj} attribute {key} is not a boolean")
            setattr(obj, key, not previous_value)
        else:
            setattr(obj, key_list[-1], value)

    if is_dataclass(obj):
        field_names = [field.name for field in fields(obj)]
        if key_list[-1] in field_names:
            _set(obj, key_list[-1], value)
        else:
            raise ValueError(f"field {key_list[-1]} not found in {obj}")
    else:
        _set(obj, key_list[-1], value)


def _apply_change(params, change):
    """Apply the change to the params object in place."""
    if callable(change):
        change(params)
    elif isinstance(change, (tuple, list)):
        if isinstance(change[0], str) and len(change) == 2:
            path, value = change
            _change_value(params, path, value)
        else:
            for c in change:
                _apply_change(params, c)
    else:
        raise ValueError(f"change {change} not recognized")
    return params


def make_progression_study(start_point, changes, return_cross_prod=True):
    """A kind of ablation study by changing the start_point with changes one by
    one.

    Args:
        start_point: the starting point of the study
        changes: a list of changes to make to the start_point. Each change is
            either a callable or tuple containing a string identifying the path in the object to
            change and the value to change to. ex: (".obs.use_html", True)
        return_cross_prod: return a CrossProd object or just the list

    Returns:
        A CrossProd object containing a list of objects with progressive changes
        from `start_point`. If return_cross_prod is False, then it will return a
        list.

    """
    params_list = [start_point]
    for change in changes:
        params = copy.deepcopy(params_list[-1])
        _apply_change(params, change)
        params_list.append(params)

    if return_cross_prod:
        return CrossProd(params_list)
    else:
        return params_list


def make_ablation_study(start_point, changes, return_cross_prod=True):
    """Ablation study by modifying the start_point with only one change at a
    time, and restarting from the original start_point for each configuration.

    Args:
        start_point: the starting point of the ablation study
        changes: a list of changes to make to the start_point. Each change is
            either a callable or tuple containing a string identifying the path in the object to
            change and the value to change to. ex: (".obs.use_html", True)
        return_cross_prod: return a CrossProd object or just the list

    Returns:
        A CrossProd object containing a list of objects with one change
        from `start_point`. If return_cross_prod is False, then it will return a
        list.

    """
    params_list = [start_point]
    for change in changes:
        params = copy.deepcopy(start_point)
        _apply_change(params, change)
        params_list.append(params)

    if return_cross_prod:
        return CrossProd(params_list)
    else:
        return params_list


if __name__ == "__main__":
    from agentlab.agents.dynamic_prompting import Flags

    study = make_progression_study(
        start_point=Flags(),
        changes=[
            ("use_thinking", True),
            (".use_plan", True),
            (".use_criticise", True),
        ],
        return_cross_prod=True,
    )

    study = expand_cross_product(study)
    for p in study:
        print(p)

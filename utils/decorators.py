from abc import ABC, abstractmethod
from functools import update_wrapper, partial

import torch


class Decorator(ABC):
    def __init__(self, f):
        self.func = f
        update_wrapper(self, f, updated=[])  # updated=[] so that 'self' attributes are not overwritten

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def __get__(self, instance, owner):
        new_f = partial(self.__call__, instance)
        update_wrapper(new_f, self.func)
        return new_f


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    elif isinstance(x, list):
        return [to_numpy(_) for _ in x]
    else:
        return x


# noinspection PyPep8Naming
class input_to_numpy(Decorator):
    def __call__(self, *args, **kwargs):
        new_args = [to_numpy(arg) for arg in args]
        new_kwargs = {key: to_numpy(value) for key, value in kwargs.items()}
        return self.func(*new_args, **new_kwargs)

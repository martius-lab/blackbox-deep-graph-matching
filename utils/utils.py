from itertools import combinations as comb
import torch
import collections
import json
from copy import deepcopy
import ast
from warnings import warn
import sys
import time
import os

JSON_FILE_KEY = 'default_json'

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for index, (t, m, s) in enumerate(zip(tensor, self.mean, self.std)):
            tensor[index] = t * s + m
        return tensor


def n_and_l_iter_parallel(n, l, enum=False):
    def lexico_iter_list(lex_list):
        for lex in lex_list:
            yield lexico_iter(lex)
        if enum:
            yield lexico_iter(range(len(lex_list[0])))

    for zipped in zip(*n, *lexico_iter_list(l)):
        yield zipped


def lexico_iter(lex):
    return comb(lex, 2)


def torch_to_numpy_list(list_of_tensors):
    return [x.cpu().detach().numpy() for x in list_of_tensors]


def numpy_to_torch_list(list_of_np_arrays, device, dtype):
    return [torch.from_numpy(x).to(dtype).to(device) for x in list_of_np_arrays]


class ParamDict(dict):
  """ An immutable dict where elements can be accessed with a dot"""
  __getattr__ = dict.__getitem__

  def __delattr__(self, item):
    raise TypeError("Setting object not mutable after settings are fixed!")

  def __setattr__(self, key, value):
    raise TypeError("Setting object not mutable after settings are fixed!")

  def __setitem__(self, key, value):
    raise TypeError("Setting object not mutable after settings are fixed!")

  def __deepcopy__(self, memo):
    """ In order to support deepcopy"""
    return ParamDict([(deepcopy(k, memo), deepcopy(v, memo)) for k, v in self.items()])

  def __repr__(self):
    return json.dumps(self, indent=4, sort_keys=True)

def recursive_objectify(nested_dict):
  "Turns a nested_dict into a nested ParamDict"
  result = deepcopy(nested_dict)
  for k, v in result.items():
    if isinstance(v, collections.Mapping):
      result[k] = recursive_objectify(v)
  return ParamDict(result)


class SafeDict(dict):
  """ A dict with prohibiting init from a list of pairs containing duplicates"""
  def __init__(self, *args, **kwargs):
    if args and args[0] and not isinstance(args[0], dict):
      keys, _ = zip(*args[0])
      duplicates =[item for item, count in collections.Counter(keys).items() if count > 1]
      if duplicates:
        raise TypeError("Keys {} repeated in json parsing".format(duplicates))
    super().__init__(*args, **kwargs)

def load_json(file):
  """ Safe load of a json file (doubled entries raise exception)"""
  with open(file, 'r') as f:
    data = json.load(f, object_pairs_hook=SafeDict)
  return data


def update_recursive(d, u, defensive=False):
  for k, v in u.items():
    if defensive and k not in d:
      raise KeyError("Updating a non-existing key")
    if isinstance(v, collections.Mapping):
      d[k] = update_recursive(d.get(k, {}), v)
    else:
      d[k] = v
  return d

def is_json_file(cmd_line):
  try:
    return os.path.isfile(cmd_line)
  except Exception as e:
    warn('JSON parsing suppressed exception: ', e)
    return False


def is_parseable_dict(cmd_line):
  try:
    res = ast.literal_eval(cmd_line)
    return isinstance(res, dict)
  except Exception as e:
    warn('Dict literal eval suppressed exception: ', e)
    return False

def update_params_from_cmdline(cmd_line=None, default_params=None, custom_parser=None, verbose=True):
  """ Updates default settings based on command line input.

  :param cmd_line: Expecting (same format as) sys.argv
  :param default_params: Dictionary of default params
  :param custom_parser: callable that returns a dict of params on success
  and None on failure (suppress exceptions!)
  :param verbose: Boolean to determine if final settings are pretty printed
  :return: Immutable nested dict with (deep) dot access. Priority: default_params < default_json < cmd_line
  """
  if not cmd_line:
    cmd_line = sys.argv

  if default_params is None:
    default_params = {}

  if len(cmd_line) < 2:
    cmd_params = {}
  elif custom_parser and custom_parser(cmd_line):  # Custom parsing, typically for flags
    cmd_params = custom_parser(cmd_line)
  elif len(cmd_line) == 2 and is_json_file(cmd_line[1]):
    cmd_params = load_json(cmd_line[1])
  elif len(cmd_line) == 2 and is_parseable_dict(cmd_line[1]):
    cmd_params = ast.literal_eval(cmd_line[1])
  else:
    raise ValueError('Failed to parse command line')

  update_recursive(default_params, cmd_params)

  if JSON_FILE_KEY in default_params:
    json_params = load_json(default_params[JSON_FILE_KEY])
    if 'default_json' in json_params:
      json_base = load_json(json_params[JSON_FILE_KEY])
    else:
      json_base = {}
    update_recursive(json_base, json_params)
    update_recursive(default_params, json_base)

  update_recursive(default_params, cmd_params)
  final_params = recursive_objectify(default_params)
  if verbose:
    print(final_params)

  update_params_from_cmdline.start_time = time.time()
  return final_params

update_params_from_cmdline.start_time = None
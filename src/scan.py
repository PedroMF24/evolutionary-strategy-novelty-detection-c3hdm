# %%
import os
from warnings import simplefilter

import pandas as pd

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


import yaml

from algorithms import all_samplers
from penalties import all_penalties

# %%
defaults = yaml.safe_load(open("defaults.yml", "r"))
if os.path.isfile("defaults-local.yml"):
    defaults_local = yaml.safe_load(open("defaults-local.yml", "r"))
    if defaults_local:
        for k, v in defaults_local.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    defaults[k][kk] = vv
            else:
                defaults[k] = v

# This is quite ugly but I can't think of a better way of setting an unique experiment name
experiment_name = os.environ.get("experiment_name", defaults["experiment_name"])
episode_name = os.environ.get("episode_name", "dev")
defaults["experiment_name"] = experiment_name
defaults["episode_name"] = episode_name
aim_path = "aim"
output_path = os.path.join("data", experiment_name, episode_name)
os.makedirs(output_path, exist_ok=True)
sampler = defaults["sampler"]
penalty = defaults["penalty"]["parameter"]["model"]

assert sampler in all_samplers.keys()
assert penalty in all_penalties.keys() or penalty is None

if os.path.exists("3HDM.in"):
    os.remove("3HDM.in")  #

if defaults["HT"]:
    os.symlink("3HDM-HT.in", "3HDM.in")
else:
    os.symlink("3HDM-No-HT.in", "3HDM.in")

all_samplers[sampler](defaults=defaults)

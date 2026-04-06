# %%
import os
import yaml
import json
import glob

import pandas as pd
import numpy as np
import docker
from multiprocessing import Pool, cpu_count

client = docker.from_env()
host_working_dir = os.getcwd()
uid = os.getuid()
gid = os.getgid()

defaults = yaml.safe_load(open("configs/defaults.yml", "r"))
defaults_local = yaml.safe_load(open("defaults-local.yml", "r"))
if defaults_local:
    for k, v in defaults_local.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                defaults[k][kk] = vv
        else:
            defaults[k] = v

experiment_name = defaults["experiment_name"]
output_path = os.path.join("data", experiment_name)

# %%

all_points_files = glob.glob(os.path.join(output_path, "*/", "all_points.csv.gz"))
all_good_points_files = glob.glob(os.path.join(output_path, "*/", "all_good_points.csv.gz"))
all_logbooks_files = glob.glob(os.path.join(output_path, "*/", "all_logbooks.csv.gz"))


def process_point_file(path):
    df = pd.read_csv(path)
    df["episode_name"] = path.split("/")[-2]
    return df


def process_loogbook_file(path):
    df = pd.read_csv(path)
    df["episode_name"] = path.split("/")[-2]
    return df


# if len(all_points_files) > 0:
#     all_points = pd.concat(
#         [process_point_file(_file) for _file in all_points_files], ignore_index=True
#     )
#     all_points.to_parquet(f"{output_path}/all_points.parquet", index=False)

if len(all_good_points_files) > 0:
    all_good_points = pd.concat(
        [process_point_file(_file) for _file in all_good_points_files], ignore_index=True
    )
    all_good_points.to_parquet(f"{output_path}/all_good_points.parquet", index=False)

if len(all_logbooks_files) > 0:
    all_logbooks = pd.concat(
        [process_loogbook_file(_file) for _file in all_logbooks_files], ignore_index=True
    )
    all_logbooks.to_parquet(f"{output_path}/all_logbooks.parquet", index=False)

# %%

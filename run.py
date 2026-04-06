import datetime
import glob
import json
import os
import shutil
from multiprocessing import Pool, cpu_count

import docker
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

client = docker.from_env()
host_working_dir = os.getcwd()
uid = os.getuid()
gid = os.getgid()

defaults = yaml.safe_load(open("configs/defaults.yml", "r"))
if os.path.isfile("defaults-local.yml"):
    defaults_local = yaml.safe_load(open("defaults-local.yml", "r"))
    if defaults_local:
        for k, v in defaults_local.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    defaults[k][kk] = vv
            else:
                defaults[k] = v

ct = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
experiment_name = "{}-{}".format(ct, defaults["experiment_name"])
output_path = os.path.join("data", experiment_name)
n_scans = defaults["n_scans"]
n_cpus = min(n_scans, defaults["n_cpus"])


def collect_files(input_files_list, output_file):
    writer = None
    for input_file in input_files_list:
        input_file_df = pd.read_csv(input_file)
        table = pa.Table.from_pandas(input_file_df)
        if writer == None:
            writer = pq.ParquetWriter(output_file, table.schema)
        writer.write_table(table)
    writer.close()


def run_container(episode_name):
    try:
        volumes = [
            f"{host_working_dir}/data:/app/data",
            f"{host_working_dir}/aim:/app/aim",
        ]

        if os.path.exists(f"{host_working_dir}/defaults-local.yml"):
            volumes.append(
                f"{host_working_dir}/defaults-local.yml:/app/defaults-local.yml"
            )
        if os.path.exists(f"{host_working_dir}/constraints-bounds-local.yml"):
            volumes.append(
                f"{host_working_dir}/constraints-bounds-local.yml:/app/constraints-bounds-local.yml"
            )
        if os.path.exists(f"{host_working_dir}/parameter-bounds-local.yml"):
            volumes.append(
                f"{host_working_dir}/parameter-bounds-local.yml:/app/parameter-bounds-local.yml"
            )
        if isinstance(defaults["cmaes"]["centroid_seed"], str):
            volumes.append(
                f"{host_working_dir}/{defaults['cmaes']['centroid_seed']}:/app/centroid_seeds.parquet"
            )

        client.containers.run(
            "3hdm-ml:dev",
            remove=True,
            detach=False,
            name=f"{experiment_name}-{episode_name}",
            user=f"{uid}:{gid}",
            environment={
                "experiment_name": experiment_name,
                "episode_name": episode_name,
            },
            volumes=volumes,
            nano_cpus=int(1e9),
        )
    except Exception as e:
        print(f"Running container failed. Error: {e}")


with Pool(n_cpus) as p:
    workers = []
    for idx in range(n_scans):
        workers.append(p.apply_async(run_container, (idx,)))
    _ = [w.get() for w in workers]

json.dump(defaults, open(f"{output_path}/defaults.json", "w"))

all_points_files = glob.glob(os.path.join(output_path, "*", "points.csv"))
all_good_points_files = glob.glob(os.path.join(output_path, "*", "good_points.csv"))
all_logbooks_files = glob.glob(os.path.join(output_path, "*", "logbook.parquet"))
all_temp_folders = glob.glob(os.path.join(output_path, "*/"))

if len(all_points_files) > 0 and defaults["collect_all_points"]:
    # all_points = pd.concat(
    #     [pd.read_csv(_file) for _file in all_points_files], ignore_index=True
    # )
    # all_points.to_parquet(f"{output_path}/all_points.parquet", index=False)
    collect_files(all_points_files, f"{output_path}/all_points.parquet")
    for f in all_points_files:
        os.remove(f)

if len(all_good_points_files) > 0 and defaults["collect_all_good_points"]:
    # all_good_points = pd.concat(
    #     [pd.read_csv(_file) for _file in all_good_points_files], ignore_index=True
    # )
    # all_good_points.to_parquet(f"{output_path}/all_good_points.parquet", index=False)
    collect_files(all_good_points_files, f"{output_path}/all_good_points.parquet")
    for f in all_good_points_files:
        os.remove(f)

if len(all_logbooks_files) > 0 and defaults["collect_all_logbooks"]:
    all_logbooks = pd.concat(
        [pd.read_parquet(_file) for _file in all_logbooks_files], ignore_index=True
    )
    all_logbooks.to_parquet(f"{output_path}/all_logbooks.parquet", index=False)
    for f in all_logbooks_files:
        os.remove(f)

if len(all_temp_folders) > 0 and defaults["delete_all_tmp_files"]:
    for fldr in all_temp_folders:
        shutil.rmtree(fldr)

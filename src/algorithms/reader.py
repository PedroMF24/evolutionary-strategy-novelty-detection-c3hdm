import time

import numpy as np
import pandas as pd
import sys, os

import subprocess



from warnings import simplefilter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

from aim import Run
from deap import base, creator, tools
from utils.constraints import constraint_columns, constraint_HT_columns
from utils.data import goodpoint_columns
from utils.parameters import get_box_dataframe, parameter_columns
from utils.process_points import evaluate_individuals
from utils.process_points import evaluate_population_batch
from utils.process_points import evaluate_file


from utils.utils import process_metrics, save_files

np.random.seed()

N_PARAMETERS = len(parameter_columns) - 1


def filter_and_save(df, output_file):
    initial_count = len(df)

    filtered_df = df.query("GoodPointNew == 1 and GoodPoint == 1")

    kept_count = len(filtered_df)

    print(f"Initial points: {initial_count}")
    print(f"Good points: {kept_count}")
    print(f"Good point Ratio: {kept_count/initial_count *100:.2f}%")
    
    filtered_df.to_csv(output_file, index=False)

    return None

def reader(
    defaults,
):

    start_time = time.time()

    if defaults["experiment_name"]:
        run = Run(experiment=defaults["experiment_name"], repo="aim")
        hypars = {
            "n_generations": defaults["n_generations"],
            # "n_population": defaults["rs"]["n_population"],
            "sampler": "reader",
        }
        run["hparams"] = hypars
    else:
        run = None

    all_constraint_columns = constraint_columns
    if defaults["HT"]:
        all_constraint_columns += constraint_HT_columns
    if defaults["verbose"]:
        print("All constraint columns: ", all_constraint_columns)


    IN_PARAM_FILE = "init_pars.dat"
    ALL_POINTS_FILE = "inspect_points_reader.csv"
    GOOD_POINTS_FILE = "good_points.csv"
    

    if not os.path.exists(IN_PARAM_FILE):
        print(f"Error: File {IN_PARAM_FILE} not found")
        sys.exit(1)

    # param_data = pd.read_csv(IN_PARAM_FILE, sep="\s+", header=None, engine="python", dtype=float)

    results = evaluate_file(IN_PARAM_FILE, 
            all_constraint_columns=all_constraint_columns,
            defaults=defaults)
    
    results.to_csv(f'data/{ALL_POINTS_FILE}', index=False, sep=',')
    filter_and_save(results, f'data/{GOOD_POINTS_FILE}')

    print("") 
    print("To retrieve to files from docker, run:")
    print(f"docker cp <name of container>:/app/data/{GOOD_POINTS_FILE} data/{GOOD_POINTS_FILE}")
    print("Check container name under CONTAINER ID aftrer running:")
    print("$docker ps -a")
    print("") 
    print("To attribute a name to a container run:")
    print("$docker build -t <name of image> .")
    print("$docker run  -it --name <name of container> <name of image>")     


    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Execution time: {elapsed_time:.3f} seconds")
    
    print("-- End program --")
    return None, None

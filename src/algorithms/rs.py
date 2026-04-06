import time

import numpy as np
import pandas as pd

from warnings import simplefilter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

from aim import Run
from deap import base, creator, tools
from utils.constraints import constraint_columns, constraint_HT_columns
from utils.data import goodpoint_columns
from utils.parameters import get_box_dataframe, parameter_columns
from utils.process_points import evaluate_individuals
from utils.utils import process_metrics, save_files

np.random.seed()

N_PARAMETERS = len(parameter_columns) - 1


def rs(
    defaults,
):
    time_total_start = time.time()
    if defaults["experiment_name"]:
        run = Run(experiment=defaults["experiment_name"], repo="aim")
        hypars = {
            "n_generations": defaults["n_generations"],
            "n_population": defaults["rs"]["n_population"],
            "sampler": "rs",
        }
        run["hparams"] = hypars
    else:
        run = None

    all_constraint_columns = constraint_columns
    if defaults["HT"]:
        all_constraint_columns += constraint_HT_columns
    if defaults["verbose"]:
        print("All constraint columns: ", all_constraint_columns)

    creator.create(
        "FitnessMin",
        base.Fitness,
        weights=(-1.0,),
    )
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()

    def init_individual_parameter():
        return np.random.random()

    toolbox.register("initIndividual", init_individual_parameter)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.initIndividual,
        n=N_PARAMETERS,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    logbook = tools.Logbook()
    header = [
        "gen",
        "time_execution",
        "time_generation",
        "n_candidates",
        "mean_valid_constraints",
        "mean_constraints",
        "min_max_constraints",
        "good_point_new_mean",
        "density_penalty_mean",
    ] + goodpoint_columns
    if defaults["HT"]:
        header.extend("GoodHB")

    logbook.header = header
    counter_good_points = 0

    for idx in range(0, defaults["n_generations"]):
        generation_start = time.time()
        population = toolbox.population(n=defaults["rs"]["n_population"])
        execution_start = time.time()
        population, results = evaluate_individuals(
            individuals=population,
            all_constraint_columns=all_constraint_columns,
            defaults=defaults,
        )
        time_execution = time.time() - execution_start
        # Meta tracking
        fitnesses = []
        for _child in population:
            fitnesses.append(_child.fitness.values)
        results["GoodPointNew"] = (
            (results[all_constraint_columns] == 0).all(1).astype(int)
        )
        results["generation"] = idx
        population_box = get_box_dataframe(population=population)
        results = pd.merge(population_box, results, left_index=True, right_index=True)
        counter_good_points += results.query("GoodPointNew == 1").shape[0]
        save_files(defaults=defaults, results=results)
        time_generation = time.time() - generation_start
        time_total = time.time() - time_total_start
        process_metrics(
            results=results,
            logbook=logbook,
            run=run,
            defaults=defaults,
            **{
                "time_generation": time_generation,
                "time_execution": time_execution,
                "time_total": time_total,
                "time_overhead": time_generation - time_execution,
                "fitness_mean": np.mean(fitnesses),
                "fitness_max": np.max(fitnesses),
                "fitness_min": np.min(fitnesses),
                "fintess_median": np.median(fitnesses),
            },
        )
        if defaults["verbose"]:
            print(logbook.stream)
        # Finish the generation
        if counter_good_points > defaults["early_stop_n_valid_points"]:
            print(
                "Eary stopping as number of valid points reached {}.".format(
                    defaults["early_stop_n_valid_points"]
                )
            )
            break

    logbook_df = pd.DataFrame(logbook)
    save_files(defaults=defaults, logbook=logbook_df)
    return None, None

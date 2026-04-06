import time

import numpy as np
import pandas as pd
from cmaes import CMA

# from deap import base, cma, creator, tools
from deap import tools
from sklearn.preprocessing import MinMaxScaler, minmax_scale

from aim import Run
from penalties import all_penalties
from utils.constraints import (
    constraint_a1_b1_repulsion,
    constraint_columns,
    constraint_HT_columns,
    NUMERICAL_INF_LOG,
)
from utils.data import HT_columns, goodpoint_columns, observable_columns, mass_columns
from utils.parameters import get_box_dataframe, parameter_box_columns, parameter_columns
from utils.process_points import (
    # evaluate_individuals,
    evaluate_population_batch,
    penalty_columns,
)
from utils.utils import process_metrics, save_files

N_PARAMETERS = len(parameter_columns) - 1  # Remove one for the higgs mass
N_CONSTRAINTS = 1
BOUND_LOW, BOUND_UP = 0.0, 1.0


def cmaes2(
    defaults,
):
    np.random.seed()
    time_total_start = time.time()

    if defaults["cmaes"]["sigma"]:
        sigma0 = defaults["cmaes"]["sigma"]
    else:
        sigma0 = 1 / np.sqrt(N_PARAMETERS)

    if isinstance(defaults["cmaes"]["centroid_seed"], str):
        centroid_seed = pd.read_parquet("centroid_seeds.parquet")
    elif isinstance(defaults["cmaes"]["centroid_seed"], list):
        centroid_seed = defaults["cmaes"]["centroid_seed"]

    if defaults["experiment_name"]:
        run = Run(experiment=defaults["experiment_name"], repo="aim")
        hypars = {
            "n_generations": defaults["n_generations"],
            "sampler": "cmaes",
            "sigma": sigma0,
            "centroid": defaults["cmaes"]["centroid_seed"],
            "penalty_parameter_warmup": defaults["penalty"]["parameter"]["warmup"],
            "penalty_parameter_cooldown": defaults["penalty"]["parameter"]["cooldown"],
            "penalty_parameter_model": defaults["penalty"]["parameter"]["model"],
            "penalty_observable_warmup": defaults["penalty"]["observable"]["warmup"],
            "penalty_observable_cooldown": defaults["penalty"]["observable"][
                "cooldown"
            ],
            "penalty_observable_model": defaults["penalty"]["observable"]["model"],
            "restart": defaults["restart"],
            "early_stop_n_valid_points": defaults["early_stop_n_valid_points"],
        }

        run["hparams"] = hypars
    else:
        run = None

    all_constraint_columns = constraint_columns
    mass_constraint_columns= [f"C{col}" for col in mass_columns]

    all_constraint_columns = [
        constraint
        for constraint in all_constraint_columns
        if constraint not in defaults["constraints_to_ignore"]
    ]

    if defaults["HT"]:
        all_constraint_columns += constraint_HT_columns

    if defaults["verbose"]:
        print("All constraint columns: ", all_constraint_columns)

    if len(defaults["penalty"]["parameter"]["focus"]) > 0:
        penalty_parameter_columns = defaults["penalty"]["parameter"]["focus"]
        penalty_parameter_columns = [
            p if "_box" in p else p + "_box" for p in penalty_parameter_columns
        ]
    else:
        penalty_parameter_columns = parameter_box_columns
    if len(defaults["penalty"]["observable"]["focus"]) > 0:
        penalty_observable_columns = defaults["penalty"]["observable"]["focus"]
    else:
        penalty_observable_columns = observable_columns
        if defaults["HT"]:
            penalty_observable_columns += HT_columns

    if defaults["verbose"]:
        print("Penalty parameter columns: ", penalty_parameter_columns)
        print("Penalty observable columns: ", penalty_observable_columns)

    all_good_points = pd.DataFrame(dtype=np.float64)

    logbook = tools.Logbook()
    header = [
        "gen",
        # "time_execution",
        # "time_generation",
        "time_total",
        "best_loss",
        "best_loss_counter",
        # "n_candidates",
        # "constraints_mean_valid",
        # "constraints_mean",
        # "constraints_min_max",
        "good_point_new_mean",
        "n_valid_points",
    ] + goodpoint_columns

    if defaults["HT"]:
        header.extend("GoodHB")

    logbook.header = header

    if isinstance(defaults["cmaes"]["centroid_seed"], str):
        centroid = (
            centroid_seed.sample(weights="weight", n=1)[parameter_box_columns]
            .iloc[0]
            .to_list()
        )
        print("Using provided seed for centroid from collection of points")
        print(centroid)
    elif isinstance(defaults["cmaes"]["centroid_seed"], list):
        centroid = centroid_seed
        print("Using provided seed for centroid")
        print(centroid)
    else:
        centroid = np.random.rand(N_PARAMETERS)  # .tolist()
        print("Random centroid")
        print(centroid)

    centroid = np.asanyarray(centroid)
    bounds = np.hstack([np.zeros((N_PARAMETERS, 1)), np.ones((N_PARAMETERS, 1))])
    optimizer = CMA(mean=centroid, sigma=sigma0, bounds=bounds)

    time_penalty_training = 0.0
    counter_restart = 0
    counter_good_points = 0
    counter_no_good_points = 0
    best_loss = np.inf
    best_loss_counter = 0
    best_n_valid_constraints = 0
    best_n_valid_constraints_counter = 0

    scaler = None
    penalty_parameter_cooldown = defaults["penalty"]["parameter"]["cooldown"]
    penalty_observable_cooldown = defaults["penalty"]["observable"]["cooldown"]
    penalty_parameter_density_estimator = None
    penalty_observable_density_estimator = None

    for idx in range(0, defaults["n_generations"]):
        time_generation_start = time.time()

        if counter_good_points > 0 and (
            defaults["penalty"]["parameter"]["model"] is not None
            or defaults["penalty"]["observable"]["model"] is not None
        ):
            time_penalty_training_start = time.time()
            if (
                defaults["penalty"]["parameter"]["model"] is not None
                and not penalty_parameter_cooldown
                and counter_good_points > defaults["penalty"]["parameter"]["warmup"]
            ):
                if (
                    isinstance(defaults["cmaes"]["centroid_seed"], str)
                    and defaults["penalty"]["use_seeds"]
                ):
                    all_good_points_for_penalties = pd.concat(
                        [
                            centroid_seed[penalty_parameter_columns],
                            all_good_points[penalty_parameter_columns],
                        ],
                        ignore_index=True,
                    )
                else:
                    all_good_points_for_penalties = all_good_points

                penalty_parameter_density_estimator = all_penalties[
                    defaults["penalty"]["parameter"]["model"]
                ](all_good_points_for_penalties[penalty_parameter_columns].values)

            if (
                defaults["penalty"]["observable"]["model"] is not None
                and not penalty_observable_cooldown
                and counter_good_points > defaults["penalty"]["observable"]["warmup"]
            ):
                if (
                    isinstance(defaults["cmaes"]["centroid_seed"], str)
                    and defaults["penalty"]["use_seeds"]
                ):
                    all_good_points_for_penalties = pd.concat(
                        [
                            centroid_seed[penalty_observable_columns],
                            all_good_points[penalty_observable_columns],
                        ],
                        ignore_index=True,
                    )
                else:
                    all_good_points_for_penalties = all_good_points

                penalty_observable_density_estimator = all_penalties[
                    defaults["penalty"]["observable"]["model"]
                ](all_good_points_for_penalties[penalty_observable_columns].values)

            time_penalty_training = time.time() - time_penalty_training_start
        else:
            time_penalty_training = 0.0

        offspring = [optimizer.ask() for _ in range(optimizer.population_size)]

        time_execution_start = time.time()
        # ===============================================================================
        # The logic for this part is new as `evaluate_individuals` assume that the points
        # are deap individuals with .fitness attribute
        # Instead, we call `evaluate_population_batch` directly, as it can be reused
        results = evaluate_population_batch(
            population=offspring,
            all_constraint_columns=all_constraint_columns,
            penalty_parameter_columns=penalty_parameter_columns,
            penalty_observable_columns=penalty_observable_columns,
            penalty_parameter_density_estimator=penalty_parameter_density_estimator,
            penalty_observable_density_estimator=penalty_observable_density_estimator,
            defaults=defaults,
        )
        results.replace([np.inf, -np.inf], np.nan, inplace=True)         
        results = results.apply(pd.to_numeric, errors='coerce')     
    	
        # We now have to mimic the logic of the rest of the `evaluate_individuals` without
        # making use of the .fitness attribute
        
        fitnesses_raw = results[all_constraint_columns + penalty_columns].copy()

        #if masses are not 125 < m < 1000, C's are not all 0 and we set fitness to log max
        columns_to_infinity = fitnesses_raw.columns.difference(mass_constraint_columns)

        #rows with at least one non zero in Cmh3 Cmh4 Cmh5
        non_zero_rows = (results[mass_constraint_columns] != 0).any(axis=1)

#        print("columns",columns_to_infinity)
#        print("rows",results[mass_constraint_columns])
#        print("nonzero",non_zero_rows)

        fitnesses_raw.loc[non_zero_rows, columns_to_infinity] = NUMERICAL_INF_LOG

#        print("fitness", fitnesses_raw)

        if scaler:
            fitnesses_raw[all_constraint_columns] = scaler.transform(
                fitnesses_raw[all_constraint_columns].values
            )

        else:
            fitnesses_raw[all_constraint_columns] = minmax_scale(
                fitnesses_raw[all_constraint_columns].values
            )

        fitnesses_constraints = list(
            fitnesses_raw[all_constraint_columns].itertuples(index=False, name=None)
        )

        fitnesses_penalties = list(
            fitnesses_raw[penalty_columns].itertuples(index=False, name=None)
        )


        fitnesses = []
        for fit_cons, fit_pen in zip(fitnesses_constraints, fitnesses_penalties):
            fit_cons_sum = sum(fit_cons)
            fit_pen_mean = np.mean(fit_pen)
            fit = fit_cons_sum + fit_pen_mean
            # prevent penalty from dominating the fitness
            if fit_cons_sum != 0:
                fit += 1
            # ----------
            fitnesses.append(fit)

        # Tell CMAES about the (individual, fitness) results
        solutions = [(i, f) for i, f in zip(offspring, fitnesses)]
        optimizer.tell(solutions)

        # End of adaptation
        # ===============================================================================
        time_execution = time.time() - time_execution_start

        results["GoodPointNew"] = (
            (results[all_constraint_columns] == 0).all(1).astype(int)
        )
        results["generation"] = idx
        offspring_box = get_box_dataframe(population=offspring)
        results = pd.merge(offspring_box, results, left_index=True, right_index=True)

        if results.query("GoodPointNew == 1").shape[0] > 0:
            # This is to to force the exploration to continue
            # in the global minima of the loss function
            counter_no_good_points = 0
            # ===============================================
            penalty_parameter_cooldown = False
            penalty_observable_cooldown = False
            all_good_points = pd.concat(
                [all_good_points, results.query("GoodPointNew == 1")], ignore_index=True
            )
            counter_good_points += results.query("GoodPointNew == 1").shape[0]
        else:
            counter_no_good_points += 1

        gen_best_loss = results[all_constraint_columns].sum(axis=1).min()
        if gen_best_loss < best_loss:
            best_loss = gen_best_loss
            best_loss_counter = 0
        else:
            best_loss_counter += 1

        gen_best_n_valid_constraints = (
            (results[all_constraint_columns] == 0).sum(axis=1).max()
        )
        if gen_best_n_valid_constraints > best_n_valid_constraints:
            best_n_valid_constraints = gen_best_n_valid_constraints
            best_n_valid_constraints_counter = 0
        else:
            best_n_valid_constraints_counter += 1

        save_files(defaults=defaults, results=results)

        # if scaler is None:
        #     scaler = MinMaxScaler(clip=True).fit(results[all_constraint_columns].values)
        # else:
        #     scaler.partial_fit(results[all_constraint_columns].values)

        time_generation = time.time() - time_generation_start
        time_total = time.time() - time_total_start
        process_metrics(
            results=results,
            logbook=logbook,
            run=run,
            defaults=defaults,
            **{
                "cmaes_sigma": optimizer._sigma,
                # "cmaes_mu": optimizer._mu,
                # "cmaes_lambda": optimizer.population_size,
                "n_valid_points": counter_good_points,
                "time_generation": time_generation,
                "time_execution": time_execution,
                "time_total": time_total,
                "time_penalty_training": time_penalty_training,
                "time_overhead": time_generation - time_execution,
                "counter_restart": counter_restart,
                "counter_no_good_points": counter_no_good_points,
                "best_loss": best_loss,
                "best_loss_counter": best_loss_counter,
                "best_n_valid_constraints": best_n_valid_constraints,
                "best_n_valid_constraints_counter": best_n_valid_constraints_counter,
                # "fitness_mean": np.mean(fitnesses),
                # "fitness_max": np.max(fitnesses),
                "fitness_min": np.min(fitnesses),
                # "fintess_median": np.median(fitnesses),
            },
        )
        if defaults["verbose"]:
            print(logbook.stream)

        stopping_criteria = {
            "optimizer.should_stop": optimizer.should_stop(),
            # "best_loss": best_loss_counter >= defaults["cmaes"]["best_loss_patience"],
            # "best_n_valid_constraints": best_n_valid_constraints_counter>= defaults["cmaes"]["best_n_valid_constraints_counter_patience"],
            "patience": all(
                [
                    best_loss_counter >= defaults["cmaes"]["best_loss_patience"],
                    best_n_valid_constraints_counter
                    >= defaults["cmaes"]["best_n_valid_constraints_counter_patience"],
                ]
            ),
            "early_stop_n_valid_points": counter_good_points
            >= defaults["early_stop_n_valid_points"],
        }

        if any(stopping_criteria.values()):
            print("At least one stopping criteria hit.")
            print(print(stopping_criteria))
            if counter_good_points >= defaults["early_stop_n_valid_points"]:
                print("Exiting as enough good points have been found.")
                break
            if counter_no_good_points < defaults["cmaes"]["no_good_patience_restart"]:
                print("Still finding good points. Continuing.")
                continue
            if counter_restart > defaults["early_stop_n_restarts"]:
                print("Exiting as maximum number of restarts have been achieved.")
                break
            if not defaults["restart"]:
                print("Exiting as restart is set to False.")
                break

            print("Restarting Evolutionary Strategy.")

            if isinstance(defaults["cmaes"]["centroid_seed"], str):
                centroid = (
                    centroid_seed.sample(weights="weight", n=1)[parameter_box_columns]
                    .iloc[0]
                    .to_list()
                )
            elif isinstance(defaults["cmaes"]["centroid_seed"], list):
                centroid = centroid_seed
            else:
                centroid = np.random.rand(N_PARAMETERS)

            centroid = np.asanyarray(centroid)
            optimizer = CMA(mean=centroid, sigma=sigma0, bounds=bounds)

            penalty_parameter_cooldown = defaults["penalty"]["parameter"]["cooldown"]
            penalty_parameter_density_estimator = None
            penalty_observable_cooldown = defaults["penalty"]["observable"]["cooldown"]
            penalty_observable_density_estimator = None
            counter_restart += 1
            counter_no_good_points = 0

    logbook_df = pd.DataFrame(logbook)
    save_files(defaults=defaults, logbook=logbook_df)
    return None, None

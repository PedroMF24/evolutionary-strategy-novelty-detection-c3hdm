import time

import numpy as np
import pandas as pd
from aim import Run
from deap import base, cma, creator, tools
from sklearn.preprocessing import MinMaxScaler

from penalties import all_penalties
from utils.constraints import (
    constraint_a1_b1_repulsion,
    constraint_columns,
    constraint_HT_columns,
)
from utils.data import HT_columns, goodpoint_columns, observable_columns
from utils.parameters import get_box_dataframe, parameter_box_columns, parameter_columns
from utils.process_points import evaluate_individuals
from utils.utils import process_metrics, save_files

N_PARAMETERS = len(parameter_columns) - 1  # Remove one for the higgs mass
N_CONSTRAINTS = 1
BOUND_LOW, BOUND_UP = 0.0, 1.0


def cmaes(
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

    all_constraint_columns = [
        constraint
        for constraint in all_constraint_columns
        if constraint not in defaults["constraints_to_ignore"]
    ]

    if defaults["a1_b1_repulsion"]:
        all_constraint_columns += constraint_a1_b1_repulsion
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

    creator.create(
        "FitnessMin",
        base.Fitness,
        weights=(-1.0,),
    )

    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()

    def init_individual_parameter():
        return np.random.rand()

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
        "time_total",
        "n_candidates",
        "constraints_mean_valid",
        "constraints_mean",
        "constraints_min_max",
        "good_point_new_mean",
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
        centroid = np.random.rand(N_PARAMETERS).tolist()
        print("Random centroid")
        print(centroid)

    strategy = cma.Strategy(
        centroid=centroid,
        sigma=sigma0,
    )
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    counter_no_good_points = 0
    time_penalty_training = 0.0
    counter_restart = 0
    counter_good_points = 0

    equalfunvals_k = int(np.ceil(0.1 + strategy.lambda_ / 4.0))
    equalfunvalues = []
    bestvalues = []
    medianvalues = []
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

        offspring = []
        counter = 0

        while True:
            if len(offspring) >= strategy.lambda_:
                offspring = offspring[: strategy.lambda_]
                break

            _o = toolbox.generate()
            if counter < 100:
                for child in _o:
                    _bound_child = False
                    _bound_child = all((c < 1) and (c > 0) for c in child)
                    if _bound_child:
                        offspring.append(child)
            elif counter >= 100:
                for child in _o:
                    _bound_child = False
                    _bound_child = all((c < 1) and (c > 0) for c in child)
                    if not _bound_child:
                        _new_child = child
                        _new_parameters = toolbox.individual()
                        for jdx in range(N_PARAMETERS):
                            if child[jdx] < 0 or child[jdx] > 1:
                                _new_child[jdx] = _new_parameters[jdx]
                        offspring.append(_new_child)
            counter += 1

        time_execution_start = time.time()
        offspring, results = evaluate_individuals(
            individuals=offspring,
            penalty_parameter_columns=penalty_parameter_columns,
            penalty_observable_columns=penalty_observable_columns,
            penalty_parameter_density_estimator=penalty_parameter_density_estimator,
            penalty_observable_density_estimator=penalty_observable_density_estimator,
            all_constraint_columns=all_constraint_columns,
            scaler=scaler,
            defaults=defaults,
        )
        time_execution = time.time() - time_execution_start

        toolbox.update(offspring)

        if offspring[-1].fitness == offspring[-equalfunvals_k].fitness:
            equalfunvalues.append(1)

        # Log the best and median value of this offspring
        fitnesses = []
        for _child in offspring:
            fitnesses.append(_child.fitness.values)
        bestvalues.append(offspring[-1].fitness.values)
        medianvalues.append(offspring[int(round(len(offspring) / 2.0))].fitness.values)

        results["GoodPointNew"] = (
            (results[all_constraint_columns] == 0).all(1).astype(int)
        )
        results["generation"] = idx
        offspring_box = get_box_dataframe(population=offspring)
        results = pd.merge(offspring_box, results, left_index=True, right_index=True)
        if results.query("GoodPointNew == 1").shape[0] > 0:
            counter_no_good_points = 0
            penalty_parameter_cooldown = False
            penalty_observable_cooldown = False
            all_good_points = pd.concat(
                [all_good_points, results.query("GoodPointNew == 1")], ignore_index=True
            )
            counter_good_points += results.query("GoodPointNew == 1").shape[0]
        else:
            counter_no_good_points += 1
        save_files(defaults=defaults, results=results)
        if scaler is None:
            scaler = MinMaxScaler(clip=True).fit(results[all_constraint_columns].values)
        else:
            scaler.partial_fit(results[all_constraint_columns].values)

        time_generation = time.time() - time_generation_start
        time_total = time.time() - time_total_start
        process_metrics(
            results=results,
            logbook=logbook,
            run=run,
            defaults=defaults,
            **{
                "cmaes_sigma": strategy.sigma,
                "cmaes_mu": strategy.mu,
                "n_valid_points": counter_good_points,
                "time_generation": time_generation,
                "time_execution": time_execution,
                "time_total": time_total,
                "time_penalty_training": time_penalty_training,
                "time_overhead": time_generation - time_execution,
                "counter_restart": counter_restart,
                "counter_no_good_points": counter_no_good_points,
                "fitness_mean": np.mean(fitnesses),
                "fitness_max": np.max(fitnesses),
                "fitness_min": np.min(fitnesses),
                "fintess_median": np.median(fitnesses),
            },
        )
        if defaults["verbose"]:
            print(logbook.stream)
        if counter_good_points > defaults["early_stop_n_valid_points"]:
            print(
                "Early stopping as number of valid points reached {}.".format(
                    defaults["early_stop_n_valid_points"]
                )
            )
            break
        if (
            counter_restart > defaults["early_stop_n_restarts"]
            and counter_no_good_points > 1000
        ):
            break

        if defaults["restart"]:
            _should_stop = should_stop(
                strategy=strategy,
                equalfunvalues=equalfunvalues,
                bestvalues=bestvalues,
                medianvalues=medianvalues,
                defaults=defaults,
                sigma0=sigma0,
            )
            if (
                _should_stop
                and (
                    counter_good_points == 0
                    or counter_no_good_points > defaults["cmaes"]["stagnation_recent"]
                )
            ) or counter_no_good_points > defaults["cmaes"]["no_good_patience_restart"]:
                print("Restarting Evolutionary Strategy.")

                if isinstance(defaults["cmaes"]["centroid_seed"], str):
                    centroid = (
                        centroid_seed.sample(weights="weight", n=1)[
                            parameter_box_columns
                        ]
                        .iloc[0]
                        .to_list()
                    )
                elif isinstance(defaults["cmaes"]["centroid_seed"], list):
                    centroid = centroid_seed
                else:
                    centroid = np.random.rand(N_PARAMETERS).tolist()

                strategy = cma.Strategy(
                    centroid=centroid,
                    sigma=sigma0,
                )
                toolbox.register("generate", strategy.generate, creator.Individual)
                toolbox.register("update", strategy.update)

                equalfunvals_k = int(np.ceil(0.1 + strategy.lambda_ / 4.0))
                equalfunvalues = []
                bestvalues = []
                medianvalues = []
                penalty_parameter_cooldown = defaults["penalty"]["parameter"][
                    "cooldown"
                ]
                penalty_parameter_density_estimator = None
                penalty_observable_cooldown = defaults["penalty"]["observable"][
                    "cooldown"
                ]
                penalty_observable_density_estimator = None
                counter_no_good_points = 0
                counter_restart += 1

    logbook_df = pd.DataFrame(logbook)
    save_files(defaults=defaults, logbook=logbook_df)
    return None, None


def should_stop(strategy, equalfunvalues, bestvalues, medianvalues, defaults, sigma0):
    """
    From https://github.com/DEAP/deap/blob/72c0bf56469781a76736aa0087424f9f5ccb443b/examples/es/cma_bipop.py#L90
    """
    tolx = 1e-12
    tolupsigma = 10**20
    n = strategy.dim
    t = strategy.update_count
    equalfunvals = 1.0 / 3.0

    stagnation_time_lapse_multiplier = defaults["cmaes"][
        "stagnation_time_lapse_multiplier"
    ]
    stagnation_offset = defaults["cmaes"]["stagnation_offset"]
    stagnation_recent = defaults["cmaes"]["stagnation_recent"]
    stagnation_dim_multiplier = defaults["cmaes"]["stagnation_dim_multiplier"]

    stagnation_iter = int(
        np.ceil(
            stagnation_time_lapse_multiplier * t
            + stagnation_offset
            + stagnation_recent
            + stagnation_dim_multiplier * n / strategy.lambda_
        )
    )
    noeffectaxis_index = t % n

    if t > n and sum(equalfunvalues[-n:]) / float(n) > equalfunvals:
        # In 1/3rd of the last N iterations the best and k'th best solutions are equal
        print(
            f"In a {equalfunvals} (ratio) of the last N iterations the best and k'th best solutions are equal"
        )
        return True

    if all(strategy.pc < tolx) and all(np.sqrt(np.diag(strategy.C)) < tolx):
        # All components of pc and sqrt(diag(C)) are smaller than the threshold
        print("All components of pc and sqrt(diag(C)) are smaller than the threshold")
        return True

    # Need to transfer strategy.diagD[-1]**2 from pyp/np.float64 to python
    # float to avoid OverflowError
    if strategy.sigma / sigma0 > float(strategy.diagD[-1] ** 2) * tolupsigma:
        # The sigma ratio is bigger than a threshold
        print("The sigma ratio is bigger than a threshold")
        return True

    if (
        len(bestvalues) > stagnation_iter
        and len(medianvalues) > stagnation_iter
        and np.median(bestvalues[-stagnation_recent:])
        >= np.median(
            bestvalues[-stagnation_iter : -stagnation_iter + stagnation_recent]
        )
        and np.median(medianvalues[-stagnation_recent:])
        >= np.median(
            medianvalues[-stagnation_iter : -stagnation_iter + stagnation_recent]
        )
    ):
        # Stagnation occurred
        print("Stagnation occurred")
        return True

    if strategy.cond > 10**14:
        # The condition number is bigger than a threshold
        print("The condition number is bigger than a threshold")
        return True

    if all(
        strategy.centroid
        == strategy.centroid
        + 0.1
        * strategy.sigma
        * strategy.diagD[-noeffectaxis_index]
        * strategy.B[-noeffectaxis_index]
    ):
        # The coordinate axis std is too low
        print("The coordinate axis std is too low")
        return True

    if any(
        strategy.centroid
        == strategy.centroid + 0.2 * strategy.sigma * np.diag(strategy.C)
    ):
        # The main axis std has no effect
        print("The main axis std has no effect")
        return True

    return False

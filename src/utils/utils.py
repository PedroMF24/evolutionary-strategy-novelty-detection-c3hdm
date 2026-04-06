import os

from aim import Distribution
from utils.data import (
    goodpoint_columns,
    parameter_columns,
)


def save_files(defaults, results=None, logbook=None):
    experiment_name = defaults["experiment_name"]
    episode_name = defaults["episode_name"]
    output_path = os.path.join("data", experiment_name, episode_name)
    if results is not None:
        results["experiment_name"] = experiment_name
        results["episode_name"] = episode_name
        results_file = f"{output_path}/points.csv"
        good_results_file = f"{output_path}/good_points.csv"
        results.to_csv(
            results_file,
            index=False,
            mode="a",
            header=False if os.path.exists(results_file) else True,
        )
        if results.query("GoodPointNew == 1 and GoodPoint == 1").shape[0] > 0:
            results.query("GoodPointNew == 1 and GoodPoint == 1").to_csv(
                good_results_file,
                index=False,
                mode="a",
                header=False if os.path.exists(good_results_file) else True,
            )
    if logbook is not None:
        logbook["experiment_name"] = experiment_name
        logbook["episode_name"] = episode_name
        loogbook_file = f"{output_path}/logbook.parquet"
        logbook.to_parquet(loogbook_file, index=False)


def process_metrics(results, logbook, run, defaults, **kwargs):
    mean_valid_constraints = results["ProportionValidConstraints"].mean()
    mean_constraints = results["MeanConstraints"].mean()
    min_max_constraints = results["MaxConstraint"].min()
    penalty_parameter_density_mean = results["penalty_parameter_density"].mean()
    penalty_observable_density_mean = results["penalty_observable_density"].mean()
    good_point_new_mean = results["GoodPointNew"].mean()

    record = dict(
        gen=results["generation"].iloc[0],
        n_candidates=results.shape[0],
        mean_valid_constraints=mean_valid_constraints,
        mean_constraints=mean_constraints,
        min_max_constraints=min_max_constraints,
        good_point_new_mean=good_point_new_mean,
        penalty_parameter_density_mean=penalty_parameter_density_mean,
        penalty_observable_density_mean=penalty_observable_density_mean,
        **results[goodpoint_columns].mean().to_dict(),
        **kwargs,
    )

    if defaults["HT"]:
        record["GoodHB"] = results["GoodHB"].mean()

    logbook.record(**record)

    if run:
        run.track(results["generation"].iloc[0], "gen")
        run.track(results.shape[0], "n_candidates")
        run.track(mean_valid_constraints, "constraints_mean_valid")
        run.track(mean_constraints, "constraints_mean")
        run.track(min_max_constraints, "constraints_min_max")
        run.track(good_point_new_mean, "good_point_new_mean")
        run.track(penalty_parameter_density_mean, "penalty_parameter_density_mean")
        run.track(penalty_observable_density_mean, "penalty_observable_density_mean")
        if kwargs:
            for k, v in kwargs.items():
                run.track(v, k)
        for k, v in results[goodpoint_columns].mean().to_dict().items():
            run.track(v, k)
        if defaults["HT"]:
            run.track(results["GoodHB"].mean(), "GoodHB")
        for parameter in parameter_columns:
            if parameter == "MH10":
                continue
            run.track(
                Distribution(results[parameter].values),
                name=parameter,
                context={"type": "parameter"},
            )

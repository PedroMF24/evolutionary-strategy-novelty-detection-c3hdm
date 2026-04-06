import os

import numpy as np
import pandas as pd
import yaml

parameter_columns = [
    "MH10",
    "MH20",
    "mC1",
    "mC2",
    "m12sqR",
    "m13sqR",
    "m23sqR",
    "bet1",
    "bet2",
    "theta",
    "phi",
    "alp12",
    "alp13",
    "alp14",
    "alp15",
    "alp23",
    "alp24",
    "alp25",
    "alp34",
    "alp35",
    "alp45",
]


parameter_box_columns = [
    para + "_box" for para in parameter_columns if "MH10" not in para
]


parameter_bounds = yaml.safe_load(open("parameter-bounds.yml", "r"))
for p, bs in parameter_bounds.items():
    for bk, bv in bs.items():
        parameter_bounds[p][bk] = eval(bv) if isinstance(bv, str) else bv

if os.path.exists("parameter-bounds-local.yml"):
    parameter_bounds_local = yaml.safe_load(open("parameter-bounds-local.yml", "r"))
    if parameter_bounds_local:
        for p, bs in parameter_bounds_local.items():
            for bk, bv in bs.items():
                parameter_bounds[p][bk] = eval(bv) if isinstance(bv, str) else bv

defaults = yaml.safe_load(open("defaults.yml", "r"))
if os.path.exists("defaults-local.yml"):
    defaults_local = yaml.safe_load(open("defaults-local.yml", "r"))
    if defaults_local:
        for k, v in defaults_local.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    defaults[k][kk] = vv
            else:
                defaults[k] = v


def get_box_dataframe(population):
    _df = pd.DataFrame(data=population, columns=parameter_box_columns)
    _df["MH10_box"] = 1.0
    return _df[["MH10_box"] + parameter_box_columns]


def map_from_box_to_parameter_space(df):
    _df = pd.DataFrame(dtype=np.float64, columns=parameter_columns)
    _df["MH10"] = parameter_bounds["MH10"]["fixed"] * df["MH10_box"]
    _df["MH20"] = (
        parameter_bounds["MH20"]["low"]
        + (parameter_bounds["MH20"]["up"] - parameter_bounds["MH20"]["low"])
        * df["MH20_box"]
    )

    _df["mC1"] = (
        parameter_bounds["mC1"]["low"]
        + (parameter_bounds["mC1"]["up"] - parameter_bounds["mC1"]["low"])
        * df["mC1_box"]
    )
    _df["mC2"] = (
        parameter_bounds["mC2"]["low"]
        + (parameter_bounds["mC2"]["up"] - parameter_bounds["mC2"]["low"])
        * df["mC2_box"]
    )

    _df["alp12"] = (
        parameter_bounds["alp12"]["low"]
        + (parameter_bounds["alp12"]["up"] - parameter_bounds["alp12"]["low"])
        * df["alp12_box"]
    )
    _df["alp13"] = (
        parameter_bounds["alp13"]["low"]
        + (parameter_bounds["alp13"]["up"] - parameter_bounds["alp13"]["low"])
        * df["alp13_box"]
    )
    _df["alp14"] = (
        parameter_bounds["alp14"]["low"]
        + (parameter_bounds["alp14"]["up"] - parameter_bounds["alp14"]["low"])
        * df["alp14_box"]
    )
    _df["alp15"] = (
        parameter_bounds["alp15"]["low"]
        + (parameter_bounds["alp15"]["up"] - parameter_bounds["alp15"]["low"])
        * df["alp15_box"]
    )
    _df["alp23"] = (
        parameter_bounds["alp23"]["low"]
        + (parameter_bounds["alp23"]["up"] - parameter_bounds["alp23"]["low"])
        * df["alp23_box"]
    )
    _df["alp24"] = (
        parameter_bounds["alp24"]["low"]
        + (parameter_bounds["alp24"]["up"] - parameter_bounds["alp24"]["low"])
        * df["alp24_box"]
    )
    _df["alp25"] = (
        parameter_bounds["alp25"]["low"]
        + (parameter_bounds["alp25"]["up"] - parameter_bounds["alp25"]["low"])
        * df["alp25_box"]
    )
    _df["alp34"] = (
        parameter_bounds["alp34"]["low"]
        + (parameter_bounds["alp34"]["up"] - parameter_bounds["alp34"]["low"])
        * df["alp34_box"]
    )
    _df["alp35"] = (
        parameter_bounds["alp35"]["low"]
        + (parameter_bounds["alp35"]["up"] - parameter_bounds["alp35"]["low"])
        * df["alp35_box"]
    )
    _df["alp45"] = (
        parameter_bounds["alp45"]["low"]
        + (parameter_bounds["alp45"]["up"] - parameter_bounds["alp45"]["low"])
        * df["alp45_box"]
    )

    _df["theta"] = (
        parameter_bounds["theta"]["low"]
        + (parameter_bounds["theta"]["up"] - parameter_bounds["theta"]["low"])
        * df["theta_box"]
    )
    _df["phi"] = (
        parameter_bounds["phi"]["low"]
        + (parameter_bounds["phi"]["up"] - parameter_bounds["phi"]["low"])
        * df["phi_box"]
    )


    if defaults["parameter_space"]["tanbeta"]:
        _df["bet1"] = np.arctan(
            parameter_bounds["tanb1"]["low"]
            + (parameter_bounds["tanb1"]["up"] - parameter_bounds["tanb1"]["low"])
            * df["bet1_box"]
        )
        _df["bet2"] = np.arctan(
            parameter_bounds["tanb2"]["low"]
            + (parameter_bounds["tanb2"]["up"] - parameter_bounds["tanb2"]["low"])
            * df["bet2_box"]
        )

    else:
        _df["bet1"] = (
            np.arctan(parameter_bounds["tanb1"]["low"])
            + (
                np.arctan(parameter_bounds["tanb1"]["up"])
                - np.arctan(parameter_bounds["tanb1"]["low"])
            )
            * df["bet1_box"]
        )
        _df["bet2"] = (
            np.arctan(parameter_bounds["tanb2"]["low"])
            + (
                np.arctan(parameter_bounds["tanb2"]["up"])
                - np.arctan(parameter_bounds["tanb2"]["low"])
            )
            * df["bet2_box"]
        )

    m12sqR_tmp = df["m12sqR_box"] - 0.5
    m12sqR_tmp_sign = np.sign(m12sqR_tmp)
    m12sqR_tmp = (
        parameter_bounds["m12sqR"]["exp_low"]
        + (parameter_bounds["m12sqR"]["exp_up"] - parameter_bounds["m12sqR"]["exp_low"])
        * np.abs(m12sqR_tmp)
        * 2
    )
    _df["m12sqR"] = m12sqR_tmp_sign * 10**m12sqR_tmp

    m13sqR_tmp = df["m13sqR_box"] - 0.5
    m13sqR_tmp_sign = np.sign(m13sqR_tmp)
    m13sqR_tmp = (
        parameter_bounds["m13sqR"]["exp_low"]
        + (parameter_bounds["m13sqR"]["exp_up"] - parameter_bounds["m13sqR"]["exp_low"])
        * np.abs(m13sqR_tmp)
        * 2
    )
    _df["m13sqR"] = m13sqR_tmp_sign * 10**m13sqR_tmp

    m23sqR_tmp = df["m23sqR_box"] - 0.5
    m23sqR_tmp_sign = np.sign(m23sqR_tmp)
    m23sqR_tmp = (
        parameter_bounds["m23sqR"]["exp_low"]
        + (parameter_bounds["m23sqR"]["exp_up"] - parameter_bounds["m23sqR"]["exp_low"])
        * np.abs(m23sqR_tmp)
        * 2
    )
    _df["m23sqR"] = m23sqR_tmp_sign * 10**m23sqR_tmp

    return _df

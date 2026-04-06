import os

import numpy as np
import yaml

from .data import observable_columns, unitarity_columns

# Used for comparison in C
NUMERICAL_INF = np.inf
# Used as infinite penalty
NUMERICAL_INF_LOG = np.log(np.finfo(np.float64).max) + 1
EPS = np.finfo(np.float64).eps

constraints_bounds = yaml.safe_load(open("constraints-bounds.yml", "r"))
for k, v in constraints_bounds.items():
    constraints_bounds[k] = eval(v) if isinstance(v, str) else v


if os.path.exists("constraints-bounds-local.yml"):
    constraints_bounds_local = yaml.safe_load(open("constraints-bounds-local.yml", "r"))
    if constraints_bounds_local:
        for k, v in constraints_bounds_local.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    constraints_bounds[k][kk] = eval(vv) if isinstance(vv, str) else vv
            else:
                constraints_bounds[k] = eval(v) if isinstance(v, str) else v


np.random.seed()


def C(O, O_LB, O_UB):
    #    print(O, O_LB, O_UB)
    vC = np.vectorize(lambda x: max(0.0, -x + O_LB, x - O_UB), otypes=[np.float64])
    return np.log(1 + vC(O))


vmin = np.vectorize(min, otypes=[np.float64])

vmax = np.vectorize(max, otypes=[np.float64])


def check_kappas(df):
    kappa_W_centre = constraints_bounds["kappa_W_centre"]
    error_W_upper = constraints_bounds["error_W_upper"]
    error_W_lower = constraints_bounds["error_W_lower"]

    kappa_Z_centre = constraints_bounds["kappa_Z_centre"]
    error_Z_upper = constraints_bounds["error_Z_upper"]
    error_Z_lower = constraints_bounds["error_Z_lower"]

    kappa_b_centre = constraints_bounds["kappa_b_centre"]
    error_b_upper = constraints_bounds["error_b_upper"]
    error_b_lower = constraints_bounds["error_b_lower"]

    kappa_t_centre = constraints_bounds["kappa_t_centre"]
    error_t_upper = constraints_bounds["error_t_upper"]
    error_t_lower = constraints_bounds["error_t_lower"]

    kappa_tau_centre = constraints_bounds["kappa_tau_centre"]
    error_tau_upper = constraints_bounds["error_tau_upper"]
    error_tau_lower = constraints_bounds["error_tau_lower"]

    kappa_W_min = kappa_W_centre - constraints_bounds["kappa_i_sigma"] * error_W_lower
    kappa_W_max = kappa_W_centre + constraints_bounds["kappa_i_sigma"] * error_W_upper

    kappa_Z_min = kappa_Z_centre - constraints_bounds["kappa_i_sigma"] * error_Z_lower
    kappa_Z_max = kappa_Z_centre + constraints_bounds["kappa_i_sigma"] * error_Z_upper

    kappa_b_min = kappa_b_centre - constraints_bounds["kappa_i_sigma"] * error_b_lower
    kappa_b_max = kappa_b_centre + constraints_bounds["kappa_i_sigma"] * error_b_upper

    kappa_b_min_WS = (
        -kappa_b_centre - constraints_bounds["kappa_i_sigma"] * error_b_lower
    )
    kappa_b_max_WS = (
        -kappa_b_centre + constraints_bounds["kappa_i_sigma"] * error_b_upper
    )

    kappa_t_min = kappa_t_centre - constraints_bounds["kappa_i_sigma"] * error_t_lower
    kappa_t_max = kappa_t_centre + constraints_bounds["kappa_i_sigma"] * error_t_upper

    kappa_tau_min = (
        kappa_tau_centre - constraints_bounds["kappa_i_sigma"] * error_tau_lower
    )
    kappa_tau_max = (
        kappa_tau_centre + constraints_bounds["kappa_i_sigma"] * error_tau_upper
    )

    CkappaW = C(df["kappaW"], kappa_W_min, kappa_W_max)

    CkappaZ = C(df["kappaW"], kappa_Z_min, kappa_Z_max)

    CkappaU = C(df["kappaU"], kappa_t_min, kappa_t_max)

    CkappaDNS = C(df["kappaD"], kappa_b_min, kappa_b_max)
    CkappaDWS = C(df["kappaD"], kappa_b_min_WS, kappa_b_max_WS)
    CkappaD = vmin(CkappaDNS, CkappaDWS)

    CkappaL = C(df["kappaL"], kappa_tau_min, kappa_tau_max)

    return CkappaW, CkappaZ, CkappaU, CkappaD, CkappaL


def check_bound_from_below(df):
    def check_positivity(x11, x22, x33, x12, x13, x23):
        mask1 = (x11 > 0.0) & (x22 > 0.0) & (x33 > 0.0)
        x12bar = np.where(mask1, np.sqrt(x11 * x22) + x12, np.nan)
        x13bar = np.where(mask1, np.sqrt(x11 * x33) + x13, np.nan)
        x23bar = np.where(mask1, np.sqrt(x22 * x33) + x23, np.nan)
        mask2 = (x12bar >= 0.0) & (x13bar >= 0.0) & (x23bar >= 0.0)
        aux = np.where(
            mask1 & mask2,
            np.sqrt(x11 * x22 * x33)
            + x12 * np.sqrt(x33)
            + x13 * np.sqrt(x22)
            + x23 * np.sqrt(x11)
            + np.sqrt(2.0 * x12bar * x13bar * x23bar),
            np.nan,
        )

        cx11 = C(x11, 0.0, NUMERICAL_INF)
        cx22 = C(x22, 0.0, NUMERICAL_INF)
        cx33 = C(x33, 0.0, NUMERICAL_INF)
        cx12bar = np.where(mask1, C(x12bar, 0.0, NUMERICAL_INF), NUMERICAL_INF_LOG)
        cx13bar = np.where(mask1, C(x13bar, 0.0, NUMERICAL_INF), NUMERICAL_INF_LOG)
        cx23bar = np.where(mask1, C(x23bar, 0.0, NUMERICAL_INF), NUMERICAL_INF_LOG)
        cxaux = np.where(mask1 & mask2, C(aux, 0.0, NUMERICAL_INF), NUMERICAL_INF_LOG)
        return cx11, cx22, cx33, cx12bar, cx13bar, cx23bar, cxaux

    A11 = 2 * df["L1"]
    A22 = 2 * df["L2"]
    A33 = 2 * df["L3"]
    A12 = df["L4"] + df["L7"]
    A13 = df["L5"] + df["L8"]
    A23 = df["L6"] + df["L9"]

    cA11, cA22, cA33, cA12bar, cA13bar, cA23bar, cAaux = check_positivity(
        A11, A22, A33, A12, A13, A23
    )

    Lpp12 = -df["L7"]
    Lpp13 = -df["L8"]
    Lpp23 = -df["L9"]

    B11 = A11
    B22 = A22
    B33 = A33
    B12 = A12 + vmin(0, Lpp12) - 2 * np.sqrt(df["L10R"] ** 2 + df["L10I"] ** 2)
    B13 = A13 + vmin(0, Lpp13) - 2 * np.sqrt(df["L11R"] ** 2 + df["L11I"] ** 2)
    B23 = A23 + vmin(0, Lpp23) - 2 * np.sqrt(df["L12R"] ** 2 + df["L12I"] ** 2)

    cB11, cB22, cB33, cB12bar, cB13bar, cB23bar, cBaux = check_positivity(
        B11, B22, B33, B12, B13, B23
    )

    return (
        cA11,
        cA22,
        cA33,
        cA12bar,
        cA13bar,
        cA23bar,
        cAaux,
        cB11,
        cB22,
        cB33,
        cB12bar,
        cB13bar,
        cB23bar,
        cBaux,
    )


def check_oblique_parameters(df):
    U = constraints_bounds["U_centre"]
    ULB = U - constraints_bounds["U_lower"]
    UUB = U + constraints_bounds["U_upper"]

    CU = C(df["U"], ULB, UUB)

    a1 = constraints_bounds["a1"]
    a2 = constraints_bounds["a2"]
    a3 = constraints_bounds["a3"]
    a4 = constraints_bounds["a4"]
    a5 = constraints_bounds["a5"]
    a6 = constraints_bounds["a6"]

    Corr = (
        a1 * df["S"] ** 2
        + a2 * df["S"] * df["T"]
        + a3 * df["T"] ** 2
        + a4 * df["S"]
        + a5 * df["T"]
        + a6
    )

    CCorr = C(Corr, 0.0, NUMERICAL_INF)

    return CU, CCorr


def check_unitariy(df):
    cevs = [C(df[e].abs(), 0.0, 8 * np.pi) for e in unitarity_columns]
    return tuple(cevs)


def check_signal_strenghts(df):
    Cs = [
        C(
            df[k],
            constraints_bounds[k + "_centre"]
            - constraints_bounds["mu_i_sigma"] * constraints_bounds[k + "_lower"],
            constraints_bounds[k + "_centre"]
            + constraints_bounds["mu_i_sigma"] * constraints_bounds[k + "_upper"],
        )
        for k in observable_columns
        if "mu_ij" in k
    ]
    return tuple(Cs)


def check_bsg(df):
    BRbsgUB = constraints_bounds["BRbsgUB"] / (
        1.0 - constraints_bounds["BRbsg_epsilon"]
    )
    BRbsgLb = constraints_bounds["BRbsgLb"] / (
        1.0 + constraints_bounds["BRbsg_epsilon"]
    )
    return C(df["BRXsgamma"], BRbsgLb, BRbsgUB)

def check_EDM(df):
    EDMUB = constraints_bounds["EDMUB"] / (
        1.0 - constraints_bounds["EDM_epsilon"]
    )
    EDMLb = -50

#    return C(df["EDM"].abs(), EDMLb, EDMUB)
    return C(np.log10(df["EDM"].abs()), EDMLb, np.log10(EDMUB))


def check_negative_masses(df):
    CMH30=C(df["MH30"], 125 , 1000)
    CMH40=C(df["MH40"], 125 , 1000)
    CMH50=C(df["MH50"], 125 , 1000)

    return (
        CMH30,
        CMH40,
        CMH50,
    )


def check_mass_difference(df):
    return C(df["mass_diff"], constraints_bounds["mass_diffLb"] , constraints_bounds["mass_diffUB"])

def check_wrongsign(df):
    return C(df["ghjbb_s(1)"], constraints_bounds["ghbb_Lb"] , constraints_bounds["ghbb_Ub"])

def check_pseudoscalar_b(df):
    return C(df["ghjbb_p(1)"], constraints_bounds["ghbb_p_Lb"] , constraints_bounds["ghbb_p_Ub"])


def check_wrongsign_tau(df):
    return C(df["ghjee_s(1)"], constraints_bounds["ghee_Lb"] , constraints_bounds["ghee_Ub"])

def check_pseudoscalar_tau(df):
    return C(df["ghjee_p(1)"], constraints_bounds["ghee_p_Lb"] , constraints_bounds["ghee_p_Ub"])

def check_circle(df):
    return C(df["ghbb_circle"], constraints_bounds["ghbbcircle_Lb"] , constraints_bounds["ghbbcircle_Ub"])

@np.errstate(all="ignore")
def check_all_constraints(df):
    (
        df["CMH30"],
        df["CMH40"],
        df["CMH50"],
    ) = check_negative_masses(df)

    (
        df["CkappaW"],
        df["CkappaZ"],
        df["CkappaU"],
        df["CkappaD"],
        df["CkappaL"],
    ) = check_kappas(df)

    (
        df["CA11"],
        df["CA22"],
        df["CA33"],
        df["CA12bar"],
        df["CA13bar"],
        df["CA23bar"],
        df["CAaux"],
        df["CB11"],
        df["CB22"],
        df["CB33"],
        df["CB12bar"],
        df["CB13bar"],
        df["CB23bar"],
        df["CBaux"],
    ) = check_bound_from_below(df)

    df["CU"], df["CCorr"] = check_oblique_parameters(df)

    df[[f"C{col}" for col in unitarity_columns]] = np.array(check_unitariy(df)).T

    df[[f"C{col}" for col in observable_columns if "mu_ij" in col]] = np.array(
        check_signal_strenghts(df)
    ).T

    df["CBRXsgamma"] = check_bsg(df)

    df["CEDM"] = check_EDM(df)

    df["mass_diff"] = df[['MH20', 'MH30', 'MH40', 'MH50']].min(axis=1)-125
    df["Cmass_diff"]= check_mass_difference(df)
    df["Cghbb_s"]= check_wrongsign(df)
    df["Cghbb_p"]= check_pseudoscalar_b(df)
    
    df["Cghee_s"]= check_wrongsign_tau(df)
    df["Cghee_p"]= check_pseudoscalar_tau(df)
    df["ghbb_circle"] = np.sqrt(df["ghjbb_s(1)"]*df["ghjbb_s(1)"]+df["ghjbb_p(1)"]*df["ghjbb_p(1)"])
    df["Cghbb_circle"]= check_circle(df)


def check_a1_b1_repulsion(df):
    C_b1_a1_lower = C(
        df["alp1"] / (df["bet1"] + EPS),
        -NUMERICAL_INF,
        constraints_bounds["b1_a1_centre"] - constraints_bounds["b1_a1_lower"],
    )


def check_HT(df):
    #    print(df["selLim_h1_obsRatio"])
    df["CselLim_h1_obsRatio"] = C(df["selLim_h1_obsRatio"], -NUMERICAL_INF, 1)
    df["CselLim_h2_obsRatio"] = C(df["selLim_h2_obsRatio"], -NUMERICAL_INF, 1)
    df["CselLim_h3_obsRatio"] = C(df["selLim_h3_obsRatio"], -NUMERICAL_INF, 1)
    df["CselLim_h4_obsRatio"] = C(df["selLim_h4_obsRatio"], -NUMERICAL_INF, 1)
    df["CselLim_h5_obsRatio"] = C(df["selLim_h5_obsRatio"], -NUMERICAL_INF, 1)
    df["CselLim_Hp1_obsRatio"] = C(df["selLim_Hp1_obsRatio"], -NUMERICAL_INF, 1)
    df["CselLim_Hp2_obsRatio"] = C(df["selLim_Hp2_obsRatio"], -NUMERICAL_INF, 1)
    df["Cchisqdiff"] = C(df["chisqdiff"], 0, constraints_bounds["chisq_ub"])


# check_HT
# Name: selLim_h1_obsRatio, dtype: object -inf 1


constraint_columns = (
    [
        "CMH30",  # m
        "CMH40",
        "CMH50",
        "CkappaW",  # kappa
        "CkappaZ",
        "CkappaU",
        "CkappaD",
        "CkappaL",
        "CA11",  # BFB
        "CA22",
        "CA33",
        "CA12bar",
        "CA13bar",
        "CA23bar",
        "CAaux",
        "CB11",
        "CB22",
        "CB33",
        "CB12bar",
        "CB13bar",
        "CB23bar",
        "CBaux",
        "CU",  # STU
        "CCorr",
    ]
    + [f"C{col}" for col in unitarity_columns]  # UNIT coefs paper miguel
    + [f"C{col}" for col in observable_columns if "mu_ij" in col]  # Atlas muij
    + ["CBRXsgamma"]  # BSgamma
    + ["CEDM"]  # EDM
    + ["Cmass_diff"]  # mass diff
    + ["Cghbb_s"]  # wrong sign
    + ["Cghbb_p"]  # pick pseudoscalar
    + ["Cghee_s"]  # wrong sign tau
    + ["Cghee_p"]  # pick pseudoscalar tau
    + ["Cghbb_circle"]  # fill outside
)

constraint_HT_columns = [
    "CselLim_h1_obsRatio",
    "CselLim_h2_obsRatio",
    "CselLim_h3_obsRatio",
    "CselLim_h4_obsRatio",
    "CselLim_h5_obsRatio",
    "CselLim_Hp1_obsRatio",
    "CselLim_Hp2_obsRatio",
    "Cchisqdiff",
]


constraint_a1_b1_repulsion = ["C_a1_b1"]

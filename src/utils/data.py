import fortranformat as ff
import numpy as np
import pandas as pd

from .parameters import parameter_columns


def get_dataframe_from_fortran(file, column_names=None, nrows=None):
    df = pd.read_csv(
        file,
        header=None,
        delim_whitespace=True,
        nrows=nrows,
        names=column_names,
        dtype=np.float64,
    )
    return df


derived_parameters_columns = [
    "MH30",
    "MH40",
    "MH50",
    "m12sqI",
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "L6",
    "L7",
    "L8",
    "L9",
    "L10R",
    "L10I",
    "L11R",
    "L11I",
    "L12R",
    "L12I",
]


mu_columns = [
    "mu_ij(1,1)",
    "mu_ij(1,2)",
    "mu_ij(1,3)",
    "mu_ij(1,4)",
    "mu_ij(1,5)",
    "mu_ij(1,6)",
    "mu_ij(2,1)",
    "mu_ij(2,2)",
    "mu_ij(2,3)",
    "mu_ij(2,4)",
    "mu_ij(2,5)",
    "mu_ij(2,6)",
    "mu_ij(3,1)",
    "mu_ij(3,2)",
    "mu_ij(3,3)",
    "mu_ij(3,4)",
    "mu_ij(3,5)",
    "mu_ij(3,6)",
    "mu_ij(4,1)",
    "mu_ij(4,2)",
    "mu_ij(4,3)",
    "mu_ij(4,4)",
    "mu_ij(4,5)",
    "mu_ij(4,6)",
]


STUBr_columns = [
    "S",
    "T",
    "U",
    "BRXsgamma",
    "EDM",
]


goodpoint_columns = [
    "GoodPoint",
    "GoodBFB",
    "GoodUni",
    "GoodSTU",
    "GoodMus",
    "GoodBSG",
    "GoodKappas",
    "GoodEDM",
]

unitarity_columns = [
    "absev(1)",
    "absev(2)",
    "absev(3)",
    "absev(4)",
    "absev(5)",
    "absev(6)",
    "absev(7)",
    "absev(8)",
    "absev(9)",
    "absev(10)",
    "absev(11)",
    "absev(12)",
    "absev(13)",
    "absev(14)",
    "absev(15)",
    "absev(16)",
    "absev(17)",
    "absev(18)",
    "absev(19)",
    "absev(20)",
    "absev(21)",
    "absev(22)",
    "absev(23)",
    "absev(24)",
    "absev(25)",
    "absev(26)",
    "absev(27)",
]

kappa_columns = [
    "kappaW",
    "kappaU",
    "kappaD",
    "kappaL",
]

mass_columns = [
    "MH30",
    "MH40",
    "MH50",
]

coupling_columns = [
    "ghjtt_s(1)",
    "ghjtt_p(1)",
    "ghjbb_s(1)",
    "ghjbb_p(1)",
    "ghjee_s(1)",
    "ghjee_p(1)",
]

mass_diff_column = [
    "mass_diff",
    "ghbb_circle"
]

all_columns = (
    parameter_columns
    + derived_parameters_columns
    + mu_columns
    + STUBr_columns
    + goodpoint_columns
    + unitarity_columns
    + kappa_columns
    + coupling_columns
    + mass_diff_column
)

observable_columns = mu_columns + STUBr_columns + unitarity_columns + kappa_columns + mass_columns + coupling_columns + mass_diff_column

neutralIds = [f"h{i+1}" for i in range(5)]
chargedIds = [f"Hp{i+1}" for i in range(2)]

scalarIds = neutralIds + chargedIds

HT_columns =  [f"selLim_{_id}_obsRatio" for _id in scalarIds] + ["chisqdiff"]


def save_parameters_fortran_file(parameters, filename, add_dummies=False):
    _parameters = parameters.copy()
    if add_dummies:
        _df_dummies = pd.DataFrame(
            data=np.zeros((len(_parameters), len(derived_parameters_columns))),
            columns=derived_parameters_columns,
            index=_parameters.index,
        )
        _parameters = _parameters.merge(_df_dummies, left_index=True, right_index=True)
    line_format = "(E17.8, "
    for _ in range(_parameters.shape[1] - 1):
        line_format += "E18.8, "
    line_format = line_format.strip(", ")
    line_format += ")"
    header_line = ff.FortranRecordWriter(line_format)
    Formatted_df = _parameters.apply(lambda x: header_line.write(x.values), axis=1)
    Formatted_df.to_csv(filename, index=False, header=False)

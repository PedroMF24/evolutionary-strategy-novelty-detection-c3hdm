# %%
import os

import glob
import pandas as pd

from tqdm import tqdm


from src.utils.data import save_parameters_fortran_file, parameter_columns

# %%

os.makedirs("plots", exist_ok=True)

# %%

scans = [
    "paper",
]

for scan in tqdm(scans):
    for file_path in tqdm(glob.glob(f"goodparquet_massdiff/{scan}/all_good_points.parquet")):
        loaded_points = pd.read_parquet(file_path)
        save_parameters_fortran_file(
            loaded_points[parameter_columns], f"dados/{scan}.dat", True
        )

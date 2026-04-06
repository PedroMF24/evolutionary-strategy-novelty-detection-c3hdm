# %%
import os

import glob
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tqdm import tqdm


from src.utils.data import save_parameters_fortran_file, parameter_columns

# %%

os.makedirs("plots", exist_ok=True)

# %%

def collect_files(input_files_list, output_file):
    writer = None
    for input_file in input_files_list:
        input_file_df = pd.read_csv(input_file)
        table = pa.Table.from_pandas(input_file_df)
        if writer == None:
            writer = pq.ParquetWriter(output_file, table.schema)
        writer.write_table(table)
    writer.close()

scans = [
    "2025-04-22-21-26-dev"
]

for scan in tqdm(scans):
    # print(scan)
    all_good_points_files = glob.glob(os.path.join(f"data/{scan}/", "*", "good_points.csv"))

    if not all_good_points_files:
        print("No good points were found.")
    else:
        collect_files(all_good_points_files, os.path.join(f"data/{scan}/", "all_good_points.parquet"))

#    all_good_points = pd.concat(
#        [pd.read_csv(_file) for _file in all_good_points_files], ignore_index=True
#    )
#    save_parameters_fortran_file(
#        all_good_points[parameter_columns], f"dados/{scan}.dat", True
#    )
#    all_good_points.to_parquet(os.path.join(f"data/{scan}/", "all_good_points.parquet"), index=False)



# # %%
# import os

# import glob
# import pandas as pd

# from tqdm import tqdm


# from src.utils.data import save_parameters_fortran_file, parameter_columns

# # %%

# os.makedirs("plots", exist_ok=True)

# # %%

# scans = [
#     "2025-03-03-17-45-dev",
# ]

# for scan in tqdm(scans):
#     print(scan)
#     all_good_points_files = glob.glob(os.path.join(f"data/{scan}/", "*", "good_points.csv"))
#     all_good_points = pd.concat(
#         [pd.read_csv(_file) for _file in all_good_points_files], ignore_index=True
#     )
#     save_parameters_fortran_file(
#         all_good_points[parameter_columns], f"dados/{scan}.dat", True
#     )
#     all_good_points.to_parquet(os.path.join(f"data/{scan}/", "all_good_points.parquet"), index=False)

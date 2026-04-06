import pandas as pd
import glob

# Get all parquet files in a directory
file_list = glob.glob("data/combined-tau-ws/*/all_good_points.parquet")

# Read and concatenate all files
df = pd.concat([pd.read_parquet(file) for file in file_list])

# Save the combined dataframe to a new Parquet file
df.to_parquet("combined-tau-ws.parquet", engine="pyarrow")


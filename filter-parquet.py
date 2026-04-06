import pandas as pd
import numpy as np

def filter_parquet(input_file, output_file):
    df = pd.read_parquet(input_file)
    
    # condition = (df['ghjbb_p(1)'] < 0.1) & (df['ghjee_p(1)'] < 0.1) & (df['ghjtt_p(1)'] < 0.1)
    # condition = (df['ghjbb_p(1)'] < 0.1)

    theta_tau = np.abs(np.degrees(df['ghjee_p(1)'] / df['ghjee_s(1)']))
    new_columns = pd.DataFrame({
        # "smallest_mass_difference": smallest_mass_diff,
        "theta_tau_new": theta_tau,
        # "chisq_diff": chisq_diff
    })
    df2 = pd.concat([df, new_columns], axis=1)
    condition4 = df2["theta_tau_new"] < 34

    matching_indices = df2[condition4].index
    
    if len(matching_indices) > 2:
        indices_to_drop = matching_indices[2000:]
        df_filtered = df2.drop(indices_to_drop)
    else:
        df_filtered = df2
    
    df_filtered.to_parquet(output_file, index=False)

    # print(df)
    # print(df_filtered)

    
    print(f"Filtered dataset saved to {output_file}")

filter_parquet("data/combined-tau-ws/all_good_points.parquet", "filtered_output.parquet")

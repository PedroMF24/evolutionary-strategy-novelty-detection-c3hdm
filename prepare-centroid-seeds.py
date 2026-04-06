import glob

import pandas as pd
from pyod.models.hbos import HBOS

# from pyod.models.iforest import IForest
from sklearn.ensemble import IsolationForest
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.pipeline import make_pipeline

all_good_points_files = glob.glob("data/combined-tau-ws/all_good_points.parquet")

all_good_points = pd.concat(
    [pd.read_parquet(fl) for fl in all_good_points_files],
    ignore_index=True,
)


parameter_box_columns = [col for col in list(all_good_points) if "box" in col]

columns_with_nans = all_good_points[parameter_box_columns].columns[
    all_good_points[parameter_box_columns].isna().any()
].tolist()
print("Columns with at least one NaN:", columns_with_nans)

all_good_points[parameter_box_columns] = all_good_points[parameter_box_columns].fillna(0)



ocsvm = make_pipeline(Nystroem(n_components=100, n_jobs=-1), SGDOneClassSVM()).fit(
    all_good_points[parameter_box_columns].values
)

ocsvm_weights = ocsvm.decision_function(all_good_points[parameter_box_columns].values)
ocsvm_weights = -1 * ocsvm_weights
ocsvm_weights = (ocsvm_weights - ocsvm_weights.min()) / (
    ocsvm_weights.max() - ocsvm_weights.min()
)

hbos = HBOS(n_bins=100).fit(all_good_points[parameter_box_columns].values)


hbos_weights = hbos.predict_proba(all_good_points[parameter_box_columns].values)
hbos_weights = hbos_weights[:, 1]

all_good_points["ocsvm_weights"] = ocsvm_weights
all_good_points["hbos_weights"] = hbos_weights
all_good_points["weight"] = hbos_weights

all_good_points.to_parquet("dev-centroid-seeds-ws-complete.parquet")

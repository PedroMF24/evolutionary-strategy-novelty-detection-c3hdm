import numpy as np
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.loda import LODA
from pyod.models.sampling import Sampling
from scipy.spatial.distance import cdist
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDOneClassSVM
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

# class MinDistancePenaliser:
#     def __init__(self, X, n_dim=1, m=1, alpha=1, *args, **kwargs):
#         self.X = X
#         self.alpha = alpha
#         self.n_dim = n_dim
#         self.m = m
#         distances = cdist(self.X, self.X, "minkowski", p=1)
#         distances_min = np.array(
#             np.ma.masked_array(distances, mask=distances == 0).min(1)
#         )
#         penalties = -distances_min
#         self.scaler = MinMaxScaler(clip=True).fit(penalties.reshape(-1, 1))

#     def get_penalties(self, x):
#         distances = cdist(x, self.X, "minkowski", p=1)
#         penalties = -distances.min(1)
#         normalised_penalties = self.scaler.transform(penalties.reshape(-1, 1)).reshape(
#             -1
#         )
#         return normalised_penalties


class MinDistancePenaliser:
    def __init__(self, X, p=2, *args, **kwargs):
        self.X = X
        self.p = p
        self.n_dim = self.X.shape[1]

    def get_penalties(self, x):
        distances = cdist(x, self.X, "minkowski", p=self.p).min(1)
        return 1 - distances / (self.n_dim ** (1 / self.p))


class OCSVMPenaliser:
    def __init__(self, X, gamma=0.5, nu=0.5, *args, **kwargs):
        self.X = X
        self.gamma = gamma
        self.nu = nu

        self.scaler = MinMaxScaler(clip=True)
        self.kernel = RBFSampler(gamma="scale", n_components=min(X.shape[0], 100))
        self.ocsvm = SGDOneClassSVM(nu=nu)
        self.pipeline = make_pipeline(self.kernel, self.ocsvm).fit(X)

        self.scaler.fit(self.pipeline.score_samples(X).reshape(-1, 1))

    def get_penalties(self, x, *args, **kwargs):
        scores = self.pipeline.score_samples(x)
        scores = self.scaler.transform(scores.reshape(-1, 1))
        return scores.reshape(-1)


class HBOSPenaliser:
    def __init__(self, X, n_bins=100, *args, **kwargs):
        self.X = X
        self.n_bins = n_bins

        self.hbos = HBOS(n_bins=self.n_bins).fit(X)

    def get_penalties(self, x):
        scores = self.hbos.predict_proba(x)[:, 0]
        return scores


class CBLOFPenaliser:
    def __init__(self, X, *args, **kwargs):
        self.X = X

        self.cblof = CBLOF().fit(X)

    def get_penalties(self, x):
        scores = self.cblof.predict_proba(x)[:, 0]
        return scores


class IForestPenaliser:
    def __init__(self, X, *args, **kwargs):
        self.X = X

        self.iforest = IForest().fit(X)

    def get_penalties(self, x):
        scores = self.iforest.predict_proba(x)[:, 0]
        return scores


class KNNPenaliser:
    def __init__(self, X, *args, **kwargs):
        self.X = X

        self.knn = KNN().fit(X)

    def get_penalties(self, x):
        scores = self.knn.predict_proba(x)[:, 0]
        return scores


class SamplingPenaliser:
    def __init__(self, X, *args, **kwargs):
        self.X = X

        self.sampling = Sampling(subset_size=min(20, X.shape[0])).fit(X)

    def get_penalties(self, x):
        scores = self.sampling.predict_proba(x)[:, 0]
        return scores


class LODAPenaliser:
    def __init__(self, X, n_bins="auto", *args, **kwargs):
        self.X = X

        self.n_bins = n_bins
        self.loda = LODA(n_bins=self.n_bins).fit(X)

    def get_penalties(self, x):
        scores = self.loda.predict_proba(x)[:, 0]
        return scores


all_penalties = {
    "mindistance": MinDistancePenaliser,
    "loda": LODAPenaliser,
    "sampling": SamplingPenaliser,
    "knn": KNNPenaliser,
    "iforest": IForest,
    "cblof": CBLOFPenaliser,
    "hbos": HBOSPenaliser,
    "ocsvm": OCSVMPenaliser,
}

if __name__ == "__main__":
    X = np.random.randn(10, 20)
    XX = np.random.randn(10, 20) + 0.1
    ocsvm = OCSVMPenaliser(X)
    print(ocsvm.get_penalties(X).mean())
    print(ocsvm.get_penalties(XX).mean())

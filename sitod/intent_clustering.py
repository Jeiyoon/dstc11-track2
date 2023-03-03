"""
Intent clustering interfaces and baseline code.
"""
import random

import torch
import copy
import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

import hyperopt.hp as hp
import numpy as np
# noinspection PyPackageRequirements
import umap
from allennlp.common import Registrable
from hyperopt import STATUS_OK, Trials, fmin, tpe, STATUS_FAIL
from hyperopt.early_stop import no_progress_loss
from hyperopt.pyll import scope
from numpy import ndarray, argmax
from sentence_transformers import SentenceTransformer
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN, OPTICS, Birch, BisectingKMeans, MeanShift
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from scipy.spatial import Voronoi, voronoi_plot_2d

from sitod.data import DialogueDataset

# from pyvot.vot_torch import UVWB, ICPVWB, SVWB, VotReg, VWB, RegVWB
# from pyvot.vot_torch import Vot
# from pyvot.vot_numpy import VOT
# from pyvot import utils

# from simcse import SimCSE
# from DiffCSE.diffcse import DiffCSE
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class IntentClusteringContext:
    """
    Dialogue clustering context consisting of a list of dialogues and set of target turn IDs to be labeled
    with clusters.
    """

    dataset: DialogueDataset
    intent_turn_ids: Set[str]
    # output intermediate clustering results/metadata here
    output_dir: Path = None


class IntentClusteringModel(Registrable):

    def cluster_intents(self, context: IntentClusteringContext) -> Dict[str, str]:
        """
        Assign cluster IDs to intent turns within a collection of dialogues.

        :param context: dialogue clustering context

        :return: assignment of turn IDs to cluster labels
        """
        raise NotImplementedError


@dataclass
class ClusterData:
    """
    Wrapper class for cluster labels.
    """
    clusters: List[int]


@dataclass
class ClusteringContext:
    """
    Wrapper for ndarray containing clustering inputs.
    """
    features: ndarray
    """
    here12
    """
    utt_list: List
    # output intermediate clustering results/metadata here
    output_dir: Path = None
    # dynamically inject parameters to clustering algorithm here
    parameters: Dict[str, Any] = field(default_factory=dict)


class ClusteringAlgorithm(Registrable):

    def cluster(self, context: ClusteringContext) -> ClusterData:
        """
        Predict cluster labels given a clustering context consisting of raw features and any parameters
        to dynamically pass to the clustering algorithm.
        :param context: clustering context
        :return: cluster labels
        """
        raise NotImplementedError


"""
OUR Intent Clustering MODEL HERE
OUR Intent Clustering MODEL HERE
OUR Intent Clustering MODEL HERE
OUR Intent Clustering MODEL HERE
OUR Intent Clustering MODEL HERE
"""


@ClusteringAlgorithm.register('sklearn_clustering_algorithm')
class SkLearnClusteringAlgorithm(ClusteringAlgorithm):
    CLUSTERING_ALGORITHM_BY_NAME = {
        'kmeans': KMeans,
        'dbscan': DBSCAN,
        'optics': OPTICS,
        'birch': Birch,
        'gmm': GaussianMixture,
        'speclu': SpectralClustering,
        'agglo': AgglomerativeClustering,
        'meanshift': MeanShift,
        'bisect': BisectingKMeans,
        'vot_kmeans': KMeans,
    }

    def __init__(
        self,
        clustering_algorithm_name: str,
        clustering_algorithm_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a clustering algorithm with scikit-learn `ClusterMixin` interface.
        :param clustering_algorithm_name: key for algorithm, currently supports 'kmeans', 'dbscan', and 'optics'
        :param clustering_algorithm_params: optional constructor parameters used to initialize clustering algorithm
        """
        super().__init__()
        # look up clustering algorithm by key
        self._is_vot = False
        clustering_algorithm_name = clustering_algorithm_name.lower()
        if clustering_algorithm_name not in SkLearnClusteringAlgorithm.CLUSTERING_ALGORITHM_BY_NAME:
            raise ValueError(f'Clustering algorithm "{clustering_algorithm_name}" not supported')
        if clustering_algorithm_name == 'vot_kmeans':
            self._is_vot = True
        self._constructor = SkLearnClusteringAlgorithm.CLUSTERING_ALGORITHM_BY_NAME[clustering_algorithm_name]
        if not clustering_algorithm_params:
            clustering_algorithm_params = {}
        self._clustering_algorithm_params = clustering_algorithm_params

    def cluster(self, context: ClusteringContext) -> ClusterData:
        # combine base parameters with any clustering parameters from the clustering context
        params = {**self._clustering_algorithm_params.copy(), **context.parameters}
        # initialize the clustering algorithm
        algorithm = self._constructor(**params)
        # K = algorithm.n_clusters
        # mean, cov = [0.0, 0.0], [[0.02, 0], [0, 0.02]]
        #
        # y1, y2 = np.random.multivariate_normal(mean, cov, K).T
        # y = np.stack((y1, y2), axis=1).clip(-0.99, 0.99)
        """
        predict and return cluster labels
        predict and return cluster labels
        predict and return cluster labels
        predict and return cluster labels
        predict and return cluster labels
        """
        # context.features: ndarray(1205, 300): X
        # labels: list(1205): Y
        labels = algorithm.fit_predict(context.features).tolist()

        if self._is_vot:
            # y: ndarray(# of K, 300)
            y = algorithm.cluster_centers_

            # https://seducinghyeok.tistory.com/10
            y_copy = y
            # y_copy = torch.tensor(y).double()
            # y_copy = torch.clamp(torch.from_numpy(y).double().clone().detach(),
            #                      -0.9999,
            #                      0.9999)

            # sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True)

            x_copy = context.features
            # x_copy = torch.tensor(context.features).double()
            # x_copy = context.features.double().clone().detach()
            # x_copy = torch.clamp(torch.from_numpy(context.features).double().clone().detach(),
            #                      -0.9999,
            #                      0.9999)

            """
            min-max norm
            # test1 = torch.max(x_copy)
            # test2 = torch.max(y_copy)
            """

            # vot = UVWB(y_copy, [x_copy], verbose=False)
            # vot = ICPVWB(y_copy, [x_copy], verbose=False)
            # vot = VOT(y_copy, [x_copy], verbose=False)
            vot = VOT(y_copy, [x_copy], verbose=False)
            # for reg in [0.5, 2, 1e9]
            # reg = 0.005
            # vot.cluster(lr=0.5, max_iter_h=1000, max_iter_y=1, beta=0.5, reg=reg)
            # (self, lr=0.5, max_iter_p=10, max_iter_h=3000, lr_decay=200, early_stop=-1, beta=0)
            output = vot.cluster()

            return ClusterData(vot.idx[0].tolist())
            # return ClusterData(output['idx'][0].tolist()) # VWB things
            # return ClusterData(output[0].tolist())

        return ClusterData(labels)


class ClusteringMetric(Registrable):
    default_implementation = 'sklearn_clustering_metric'

    def compute(self, cluster_labels: List[int], clustering_context: ClusteringContext) -> float:
        """
        Compute clustering validity score/intrinsic metric for a clustering (higher is better).
        :param cluster_labels: candidate clustering of inputs
        :param clustering_context: clustering context, containing clustering inputs
        :return: validity score
        """
        raise NotImplementedError


@ClusteringMetric.register('sklearn_clustering_metric')
class SklearnClusteringMetric(ClusteringMetric):
    METRIC_BY_NAME = {
        'silhouette_score': metrics.silhouette_score,
        'calinski_harabasz_score': metrics.calinski_harabasz_score,
        'davies_bouldin_score': metrics.davies_bouldin_score,
    }

    def __init__(self, metric_name: str, metric_params: Dict[str, Any] = None) -> None:
        """
        Initialize a `ClusteringMetric` based on built-in scikit-learn cluster validity metrics.
        :param metric_name: name of metric to use
        :param metric_params: any parameters to pass to validity metric
        """
        super().__init__()
        metric_name = metric_name.lower()
        if metric_name not in SklearnClusteringMetric.METRIC_BY_NAME:
            raise ValueError(f'Metric "{metric_name}" not supported')
        self._metric = SklearnClusteringMetric.METRIC_BY_NAME[metric_name]
        self._metric_params = dict(metric_params) if metric_params else {}

    def compute(
        self,
        cluster_labels: List[int],
        clustering_context: ClusteringContext,
    ) -> float:
        # skip cases where metrics may not be defined
        n_labels = len(set(cluster_labels))
        if n_labels <= 1:
            return -1
        features = clustering_context.features
        if n_labels == len(features):
            return 0
        params = self._metric_params.copy()
        return self._metric(features, cluster_labels, **params)


@ClusteringAlgorithm.register('hyperopt_tuned_clustering_algorithm')
class HyperoptTunedClusteringAlgorithm(ClusteringAlgorithm):
    NAME_TO_EXPRESSION = {
        'choice': hp.choice,
        'randint': hp.randint,
        'uniform': hp.uniform,
        'quniform': lambda *args: scope.int(hp.quniform(*args)),
        'loguniform': hp.loguniform,
        'qloguniform': lambda *args: scope.int(hp.qloguniform(*args)),
        'normal': hp.normal,
        'qnormal': lambda *args: scope.int(hp.qnormal(*args)),
        'lognormal': hp.lognormal,
        'qlognormal': lambda *args: scope.int(hp.qlognormal(*args)),
    }

    def __init__(
        self,
        clustering_algorithm: ClusteringAlgorithm,
        metric: ClusteringMetric,
        parameter_search_space: Dict[str, List[Any]],
        max_evals: int = 100,
        timeout: Optional[int] = None,
        trials_per_eval: int = 1,
        patience: int = 25,
        min_clusters: int = 5,
        max_clusters: int = 50,
        tpe_startup_jobs: int = 10,
    ) -> None:
        """
        Initialize a clustering algorithm wrapper that finds optimal clustering parameters based on
        an intrinsic cluster validity metric using hyperopt.

        :param clustering_algorithm: clustering algorithm to for which to optimize hyperparameters
        :param metric: clustering metric used to define loss
        :param parameter_search_space: parameter search space dictionary
        :param max_evals: maximum total number of trials
        :param timeout: timeout in seconds on hyperparameter search
        :param trials_per_eval: number of trials for each unique parameter setting
        :param patience: maximum number of trials to continue after no progress on loss function
        :param min_clusters: minimum number of clusters for a valid clustering
        :param max_clusters: maximum number of clusters for a valid clustering
        :param tpe_startup_jobs: number of random trials to explore search space
        """
        self._clustering_algorithm = clustering_algorithm
        self._metric = metric
        self._space = {}
        for key, value in parameter_search_space.items():
            self._space[key] = self.NAME_TO_EXPRESSION[value[0]](key, *value[1:])
        self._max_evals = max_evals
        self._timeout = timeout
        self._trials_per_eval = trials_per_eval
        self._patience = patience
        self._min_clusters = min_clusters
        self._max_clusters = max_clusters
        self._tpe_startup_jobs = tpe_startup_jobs
        # avoid verbose hyperopt logging
        loggers_to_ignore = [
            "hyperopt.tpe",
            "hyperopt.fmin",
            "hyperopt.pyll.base",
        ]
        for ignored in loggers_to_ignore:
            logging.getLogger(ignored).setLevel(logging.ERROR)

    def cluster(self, context: ClusteringContext) -> ClusterData:
        """
        context:
            - features: ndarray(1205, 300)
            - output_dir
            - parameters: dict{} <- e.g. 'n_cluster: 42'
        """
        trials = Trials()
        """ 
        results_by_params:
        {'loss': -0.019105693325400352, 
        'n_predicted_clusters': 42, 
        'status': 'ok', 
        'labels': [26, 0, 21, 26, 1, 16, 0, 12, 12, 2, 0, 1, 29, 29, 17, 39, 11, 1, 0, 32, 11, 7, 28, 11, 11, 10, 36, 36, 12, 29, 29, 11, 21, 7, 1, 27, 32, 5, 27, 1, 3, 32, 4, 12, 39, 39, 20, 20, 9, 5, 9, 9, 12, 9, 7, 41, 12, 16, 5, 9, 2, 29, 9, 16, 38, 16, 20, 1, 32, 16, 16, 12, 32, 32, 32, 12, 7, 4, 29, 16, 5, 1, 4, 16, 0, 32, 2, 32, 5, 32, 22, 39, 5, 2, 16, 5, 34, 21, 12, 12, 32, 29, 19, 29, 26, 12, 29, 26, 19, 18, 5, 34, 16, 15, 32, 2, 1, 21, 2, 39, 34, 32, 12, 32, 19, 31, 21, 0, 25, 10, 29, 36, 39, 19, 39, 39, 20, 12, 12, 16, 32, 19, 9, 7, 12, 16, 2, 7, 9, 20, 17, 5, 5, 32, 12, 16, 6, 41, 12, 0, 34, 12, 32, 12, 32, 20, 1, 32, 9, 7, 32, 3, 16, 32, 32, 24, 19, 2, 29, 1, 1, 9, 0, 32, 16, 5, 1, 39, 7, 32, 32, 19, 17, 34, 41, 7, 9, 7, 7, 3, 12, 5, 8, 9, 9, 28, 32, 5, 1, 9, 0, 30, 39, 31, 9, 5, 12, 8, 8, 5, 9, 28, 15, 9, 32, 32, 41, 32, 29, 8, 29, 13, 29, 41, 13, 41, 20, 13, 13, 7, 26, 0, 16, 35, 5, 12, 31, 25, 37, 34, 4, 4, 20, 13, 4, 41, 23, 4, 11, 4, 32, 30, 30, 4, 4, 29, 32, 29, 29, 4, 41, 29, 29, 8, 13, 11, 30, 11, 25, 12, 4, 29, 11, 4, 30, 11, 11, 29, 13, 4, 1, 15, 4, 29, 32, 29, 32, 32, 16, 4, 10, 4, 29, 41, 13, 11, 4, 11, 41, 13, 30, 25, 41, 4, 20, 13, 11, 29, 13, 4, 4, 8, 13, 41, 23, 41, 23, 29, 13, 32, 20, 29, 41, 41, 10, 4, 4, 20, 29, 13, 11, 11, 11, 0, 1, 29, 16, 16, 12, 9, 36, 19, 2, 31, 39, 7, 12, 31, 38, 1, 5, 21, 7, 41, 28, 7, 29, 1, 39, 32, 9, 32, 1, 32, 1, 1, 15, 16, 5, 31, 36, 41, 12, 32, 32, 39, 34, 12, 9, 9, 32, 9, 2, 2, 28, 12, 12, 9, 19, 39, 29, 12, 16, 12, 29, 12, 30, 12, 29, 29, 21, 1, 12, 39, 32, 28, 3, 34, 21, 1, 39, 37, 0, 31, 0, 1, 37, 36, 38, 16, 1, 9, 16, 11, 11, 21, 32, 35, 5, 32, 32, 4, 35, 11, 39, 0, 32, 0, 0, 5, 41, 39, 25, 31, 35, 20, 32, 1, 41, 41, 25, 17, 20, 13, 41, 32, 41, 11, 29, 37, 36, 4, 20, 41, 7, 4, 21, 37, 32, 36, 22, 5, 19, 35, 16, 41, 39, 9, 32, 25, 30, 32, 41, 8, 11, 16, 32, 39, 31, 7, 32, 37, 24, 32, 19, 39, 41, 38, 0, 41, 35, 18, 28, 25, 41, 7, 30, 30, 39, 12, 32, 0, 0, 4, 11, 8, 17, 4, 32, 0, 32, 16, 5, 35, 41, 36, 1, 39, 16, 5, 36, 32, 25, 41, 11, 4, 11, 0, 41, 11, 39, 28, 9, 3, 19, 19, 37, 24, 41, 37, 41, 7, 0, 30, 41, 2, 16, 5, 25, 17, 32, 12, 9, 16, 39, 25, 0, 38, 2, 41, 32, 36, 37, 39, 15, 25, 1, 36, 7, 41, 35, 32, 37, 32, 41, 35, 4, 16, 4, 21, 0, 9, 15, 41, 4, 20, 29, 32, 29, 41, 32, 30, 38, 16, 25, 5, 9, 35, 8, 32, 21, 32, 41, 22, 5, 39, 7, 1, 41, 35, 14, 21, 31, 11, 25, 17, 20, 39, 25, 5, 41, 9, 31, 17, 0, 5, 17, 0, 17, 17, 17, 17, 20, 17, 17, 32, 16, 4, 17, 32, 19, 17, 26, 0, 29, 31, 26, 32, 7, 7, 6, 32, 34, 31, 26, 1, 37, 16, 4, 8, 26, 28, 8, 8, 16, 41, 16, 7, 30, 32, 16, 32, 36, 5, 39, 9, 29, 0, 30, 25, 16, 9, 9, 36, 12, 9, 41, 2, 1, 36, 39, 37, 21, 7, 31, 9, 39, 32, 39, 20, 1, 32, 8, 41, 7, 25, 0, 16, 26, 29, 5, 17, 16, 4, 7, 1, 32, 17, 29, 12, 21, 7, 37, 0, 5, 0, 0, 19, 0, 31, 0, 39, 9, 1, 9, 2, 6, 39, 6, 0, 12, 37, 21, 12, 0, 0, 31, 17, 0, 0, 26, 0, 24, 32, 7, 11, 37, 39, 32, 17, 32, 17, 39, 32, 1, 32, 17, 15, 19, 12, 17, 39, 32, 32, 3, 39, 8, 32, 16, 19, 3, 31, 2, 29, 9, 26, 15, 0, 16, 38, 20, 1, 32, 16, 27, 12, 32, 1, 32, 32, 2, 12, 7, 29, 16, 5, 20, 8, 0, 32, 2, 31, 9, 32, 5, 32, 22, 2, 16, 5, 34, 21, 12, 12, 29, 4, 32, 41, 0, 19, 12, 29, 12, 29, 26, 19, 5, 16, 34, 32, 2, 1, 25, 21, 39, 5, 32, 32, 12, 32, 31, 19, 12, 0, 16, 5, 25, 29, 32, 39, 18, 3, 19, 39, 39, 39, 20, 12, 5, 28, 12, 20, 1, 16, 32, 12, 16, 2, 16, 9, 32, 37, 12, 32, 3, 32, 32, 12, 0, 29, 7, 32, 39, 8, 16, 9, 9, 1, 20, 20, 20, 11, 20, 21, 1, 8, 32, 29, 32, 16, 19, 9, 8, 19, 1, 5, 29, 12, 16, 16, 0, 9, 12, 0, 0, 32, 5, 9, 27, 9, 20, 7, 12, 20, 3, 5, 30, 9, 31, 3, 19, 9, 29, 3, 32, 39, 7, 32, 9, 37, 32, 32, 25, 32, 17, 36, 7, 9, 41, 39, 24, 12, 20, 5, 12, 20, 11, 0, 32, 19, 3, 5, 29, 20, 4, 16, 1, 8, 31, 3, 5, 24, 35, 3, 7, 3, 9, 7, 7, 26, 17, 1, 29, 1, 4, 24, 8, 4, 12, 1, 24, 3, 5, 32, 16, 12, 1, 32, 32, 3, 5, 7, 21, 3, 9, 9, 32, 34, 32, 1, 33, 32, 8, 26, 0, 3, 39, 9, 12, 41, 39, 36, 5, 11, 2, 0, 36, 32, 5, 20, 4, 39, 5, 26, 9, 20, 31, 32, 3, 32, 31, 5, 4, 1, 32, 21, 19, 39, 5, 11, 24, 11, 34, 11, 5, 38, 11, 19, 20, 24, 32, 9, 12, 20, 5, 5, 39, 18, 1, 32, 34, 11, 7, 12, 5, 32, 29, 7, 3, 19, 9, 36, 9, 0, 16, 7, 3, 1, 33, 15, 1, 9, 33, 11, 1, 32, 4, 16, 0, 37, 33, 9, 39, 37, 33, 5, 27, 9, 9, 11, 16, 32, 16, 7, 1, 0, 33, 26, 4, 19, 6, 9, 29, 41, 32, 32, 7, 27, 39, 9, 9, 19, 1, 33, 40, 9, 16, 27, 32, 7, 32, 3, 1, 19, 1, 13, 5, 28, 0, 9, 9, 1, 32, 9, 12, 11, 1, 33, 21, 8, 5, 20, 33, 33, 11, 19, 39, 37, 37, 25, 16, 20, 27, 9, 2], 
        'loss_variance': 0.00020156201}
        """
        results_by_params = {}

        def _objective(params):
            # params_key: '{"n_clusters": 42}'
            params_key = json.dumps(params, sort_keys=True)
            # saved n_clusters -> skip and reload it
            if params_key in results_by_params:
                # skip repeated params
                return results_by_params[params_key]

            scores = []
            labelings = []
            try:
                for seed in range(self._trials_per_eval):
                    """
                    context:
                        - features: ndarray(1205, 300) <- [[-0.14741667  0.17008534 -0.25346    ... -0.089008    0.1685041,   0.16069223], [-0.40477502  0.333215   -0.22350499 ...  0.0699574   0.27084,   0.0447245 ], [ 0.11684766  0.05021168 -0.28317836 ...  0.02239534  0.05822584,  -0.08135883], ..., [ 0.06955656  0.06661733 -0.14817232 ... -0.017159    0.02404766,  -0.20433277], [-0.08224227  0.07018194 -0.21111435 ...  0.10326014  0.07476003,   0.00877127], [-0.09866913  0.07441893 -0.27328715 ...  0.10354973  0.06586846,   0.11059206]]
                        - output_dir
                        - parameters: dict{} <- e.g. 'n_cluster: 42'
                    """
                    trial_context = replace(context, parameters=params)
                    """
                    result:
                        - clusters: [26, 0, 21, 26, 1, 16, 0, 12, 12, 2, 0, 1, 29, 29, 17, 39, 11, 1, 0, 32, 11, 7, 28, 11, 11, 10, 36, 36, 12, 29, 29, 11, 21, 7, 1, 27, 32, 5, 27, 1, 3, 32, 4, 12, 39, 39, 20, 20, 9, 5, 9, 9, 12, 9, 7, 41, 12, 16, 5, 9, 2, 29, 9, 16, 38, 16, 20, 1, 32, 16, 16, 12, 32, 32, 32, 12, 7, 4, 29, 16, 5, 1, 4, 16, 0, 32, 2, 32, 5, 32, 22, 39, 5, 2, 16, 5, 34, 21, 12, 12...]
                    """
                    result = self._clustering_algorithm.cluster(trial_context)
                    # score: 0.019105693325400352
                    score = self._metric.compute(result.clusters, context)
                    scores.append(score)
                    labelings.append(result.clusters)
            except ValueError:
                return {
                    'loss': -1,
                    'status': STATUS_FAIL
                }

            score = float(np.mean(scores))
            labels = labelings[int(argmax(scores))]
            n_predicted_clusters = len(set(labels))
            if not (self._min_clusters <= n_predicted_clusters <= self._max_clusters):
                return {
                    'loss': 1,
                    'status': STATUS_FAIL
                }

            result = {
                'loss': -score,
                'n_predicted_clusters': n_predicted_clusters,
                'status': STATUS_OK,
                'labels': labels
            }
            if len(scores) > 1:
                result['loss_variance'] = np.var(scores, ddof=1)
            results_by_params[params_key] = result
            """
            result:
                - clusters: [26, 0, 21, 26, 1, 16, 0, 12, 12, 2, 0, 1, 29, 29, 17, 39, 11, 1, 0, 32, 11, 7, 28, 11, 11, 10, 36, 36, 12, 29, 29, 11, 21, 7, 1, 27, 32, 5, 27, 1, 3, 32, 4, 12, 39, 39, 20, 20, 9, 5, 9, 9, 12, 9, 7, 41, 12, 16, 5, 9, 2, 29, 9, 16, 38, 16, 20, 1, 32, 16, 16, 12, 32, 32, 32, 12, 7, 4, 29, 16, 5, 1, 4, 16, 0, 32, 2, 32, 5, 32, 22, 39, 5, 2, 16, 5, 34, 21, 12, 12...]
                - n_predicted_clusters: 42
                - status: ok
                - labels: list(1205) <- [26, 0, 21, 26, 1, 16, 0, 12, 12, 2, 0, 1, 29, 29, 17, 39, 11, 1, 0, 32, 11, 7, 28, 11, 11, 10, 36, 36, 12, 29, 29, 11, 21, 7, 1, 27, 32, 5, 27, 1, 3, 32, 4, 12, 39, 39, 20, 20, 9, 5, 9, 9, 12, 9, 7, 41, 12, 16, 5, 9, 2, 29, 9, 16, 38, 16, 20, 1, 32, 16, 16, 12, 32, 32, 32, 12, 7, 4, 29, 16, 5, 1, 4, 16, 0, 32, 2, 32, 5, 32, 22, 39, 5, 2, 16, 5, 34, 21, 12, 12...
                - loss_variance: 0.00020156201
            """
            return result

        tpe_partial = partial(tpe.suggest, n_startup_jobs=self._tpe_startup_jobs)
        fmin(
            _objective,
            space=self._space,
            algo=tpe_partial,
            max_evals=self._max_evals,
            trials=trials,
            timeout=self._timeout,
            rstate=np.random.default_rng(42),
            early_stop_fn=no_progress_loss(self._patience)
        )
        # tsne = TSNE(n_components=2, random_state=0)
        # X_2d = tsne.fit_transform(context.features)
        X_2d = umap.UMAP().fit_transform(context.features)
        y = np.array(trials.best_trial['result']['labels']) # intent

        """
        intent induction
        """
        # qualitative = [[u, l] for u, l in zip(context.utt_list, y)]
        # qualitative.sort(key=lambda x: x[1])
        # result_dir = "/data02/jeiyoon_park/dstc11-track2-intent-induction/results/intent_table/result_2.txt"
        # f = open(result_dir, 'w')
        #
        # for q in qualitative:
        #     text = str(q[0]) + "\t" + str(q[1]) + "\n"
        #     f.write(text)
        # f.close()

        target_ids = range(len(list(set(y))))
        target_names = list(set(y))
        len_color = len(target_names)

        plt.figure(figsize=(12, 12))
        # https://www.statology.org/matplotlib-random-color/
        # colors = set(random.sample(list(mcolors.CSS4_COLORS.keys()), len_color))
        colors = {'lightskyblue', 'indigo', 'burlywood', 'olive', 'green', 'saddlebrown', 'hotpink', 'darkgoldenrod', 'darkgreen', 'crimson', 'lightgreen', 'lavenderblush', 'paleturquoise', 'royalblue', 'greenyellow', 'cornsilk', 'darkkhaki', 'pink', 'gold', 'skyblue', 'silver', 'mediumpurple', 'blue', 'navy', 'blueviolet', 'lightcoral', 'slateblue', 'darksalmon', 'firebrick', 'turquoise', 'deeppink', 'orchid', 'darkslateblue', 'mistyrose', 'mediumaquamarine', 'orangered', 'springgreen', 'lightslategray', 'salmon', 'darkolivegreen', 'mediumseagreen', 'deepskyblue'}

        centroid_list = []
        visualize_type = "scatter"
        # visualize_type = "voronoi" # scatter

        logger.info(f'Visualize type: {visualize_type}')

        if visualize_type == "scatter":
            for i, c, label in zip(target_ids, colors, target_names):
                plt.scatter(X_2d[y == target_names[i], 0],
                            X_2d[y == target_names[i], 1],
                            s=2,
                            c=c,
                            label=label)
            plt.tight_layout(pad=0.5)
            plt.axis('off')
            # plt.axis('equal')
            # plt.gca().set_xlim([-5, 20])
            # plt.gca().set_ylim([-5, 20])
            plt.show()

        if visualize_type == "voronoi":
            for i, c, label in zip(target_ids, colors, target_names):
                centroid_x = np.mean(X_2d[y == target_names[i], 0])
                centroid_y = np.mean(X_2d[y == target_names[i], 1])

                centroid_list.append([centroid_x, centroid_y])

            # voronoi
            vor = Voronoi(centroid_list)
            fig = voronoi_plot_2d(vor,
                                  # show_vertices=False,
                                  # line_colors='black',
                                  # line_width=2,
                                  # line_alpha=0.6,
                                  point_size=0.1
                                  )

            for i, c, label in zip(target_ids, colors, target_names):
                plt.scatter(X_2d[y == target_names[i], 0],
                            X_2d[y == target_names[i], 1],
                            s=2,
                            c='lightgray',
                            label=label)

            for i, c, label in zip(target_ids, colors, target_names):
                centroid_x = np.mean(X_2d[y == target_names[i], 0])
                centroid_y = np.mean(X_2d[y == target_names[i], 1])

                outlier_check = X_2d[y == target_names[i], 0].shape[0]

                if outlier_check <= 5:
                    marker = "x"
                else:
                    marker = "*"
                plt.scatter(centroid_x, centroid_y, marker=marker, color=c) # 'r'

            plt.tight_layout(pad=0.5)
            plt.axis('equal')
            plt.gca().set_xlim([-5, 20])
            plt.gca().set_ylim([-5, 20])
            plt.show()

        return ClusterData(trials.best_trial['result']['labels'])


# https://www.sbert.net/
# https://velog.io/@tobigs-nlp/Sentence-BERT-Sentence-Embeddings-using-Siamese-BERT-Networks
class SentenceEmbeddingModel(Registrable):
    def encode(self, utterances: List[str]) -> np.ndarray:
        """
        Encode a list of utterances as an array of real-valued vectors.
        :param utterances: original utterances
        :return: output encoding
        """
        raise NotImplementedError


@SentenceEmbeddingModel.register('sentence_transformers_model')
class SentenceTransformersModel(SentenceEmbeddingModel):

    def __init__(self, model_name_or_path: str) -> None:
        """
        Initialize SentenceTransformers model for a given path or model name.
        :param model_name_or_path: model name or path for SentenceTransformers sentence encoder
        """
        super().__init__()
        self._sentence_transformer = model_name_or_path

    def encode(self, utterances: List[str]) -> np.ndarray:
        # self._sentence_transformer = "princeton-nlp/sup-simcse-roberta-large"
        self._sentence_transformer = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
        # self._sentence_transformer = 'sentence-transformers/all-MiniLM-L6-v2'
        # self._sentence_transformer = 'sentence-transformers/all-MiniLM-L12-v2'
        # self._sentence_transformer = 'sentence-transformers/all-mpnet-base-v2'
        # self._sentence_transformer ='sentence-transformers/average_word_embeddings_glove.840B.300d'
        # self._sentence_transformer = "voidism/diffcse-roberta-base-trans"
        # self._sentence_transformer = "sosuke/ease-roberta-base"

        """ Sentbert """
        encoder = SentenceTransformer(self._sentence_transformer)
        """ SimCSE """
        # encoder = SimCSE(self._sentence_transformer)
        """ DiffCSE """
        # encoder = DiffCSE(self._sentence_transformer)
        """ EASE """
        # encoder = AutoModel.from_pretrained(self._sentence_transformer)
        # tokenizer = AutoTokenizer.from_pretrained(self._sentence_transformer)
        #
        # # # Set pooler.
        # pooler = lambda last_hidden, att_mask: (last_hidden * att_mask.unsqueeze(-1)).sum(1) / att_mask.sum(-1).unsqueeze(-1)
        #
        # inputs = tokenizer(utterances, padding=True, truncation=True, return_tensors="pt")
        #
        # with torch.no_grad():
        #     last_hidden = encoder(**inputs, output_hidden_states=True, return_dict=True).last_hidden_state
        #
        # # return: tensor(1205, 786)
        # return pooler(last_hidden, inputs["attention_mask"])
        return encoder.encode(utterances)


@IntentClusteringModel.register('baseline_intent_clustering_model')
class BaselineIntentClusteringModel(IntentClusteringModel):

    def __init__(
        self,
        clustering_algorithm: ClusteringAlgorithm,
        embedding_model: SentenceEmbeddingModel,
    ) -> None:
        """
        Initialize intent clustering model based on clustering utterance embeddings.
        :param clustering_algorithm: clustering algorithm applied to sentence embeddings
        :param embedding_model: sentence embedding model
        """
        super().__init__()
        self._clustering_algorithm = clustering_algorithm
        self._embedding_model = embedding_model

    def cluster_intents(self, context: IntentClusteringContext) -> Dict[str, str]:
        # collect utterances corresponding to intents
        utterances = []
        turn_ids = []
        labels = set()
        for dialogue in context.dataset.dialogues:
            for turn in dialogue.turns:
                if turn.turn_id in context.intent_turn_ids:
                    utterances.append(turn.utterance)
                    turn_ids.append(turn.turn_id)
                    labels.update(turn.intents)

        # compute sentence embeddings
        features = self._embedding_model.encode(utterances)
        # cluster sentence embeddings
        """
        here12
        """
        context = ClusteringContext(
            features,
            utterances,
            output_dir=context.output_dir,
        )
        result = self._clustering_algorithm.cluster(context)
        # map turn IDs to cluster labels
        return {turn_id: str(label) for turn_id, label in zip(turn_ids, result.clusters)}


@SentenceEmbeddingModel.register('caching_sentence_embedding_model')
class CachingSentenceEmbeddingModelSentenceTransformersModel(SentenceEmbeddingModel):

    def __init__(
        self,
        sentence_embedding_model: SentenceEmbeddingModel,
        cache_path: str,
        prefix: str,
    ) -> None:
        """
        `SentenceEmbeddingModel` wrapper that caches sentence embeddings to disk.
        :param sentence_embedding_model: wrapped sentence embedding model
        :param cache_path: path to cache sentence embeddings
        :param prefix: cache key prefix for this model
        """
        super().__init__()
        self._sentence_embedding_model = sentence_embedding_model
        self._cache_path = Path(cache_path)
        self._cache_path.mkdir(exist_ok=True, parents=True)
        self._cache_key_prefix = prefix

    def _cache_key(self, utterances: List[str]) -> Path:
        doc = '|||'.join(utterances)
        # https://wikidocs.net/122201
        return self._cache_path / f'{self._cache_key_prefix}_{hashlib.sha256(doc.encode("utf-8")).hexdigest()}.pkl'

    def encode(self, utterances: List[str]) -> np.ndarray:
        cache_path = self._cache_key(utterances)
        if cache_path.exists():
            logger.info(f'Sentence encoder cache hit for {len(utterances)} utterances')
            with open(cache_path, "rb") as fin:
                stored_data = pickle.load(fin)
                stored_sentences = stored_data['sentences']
                stored_embeddings = stored_data['embeddings']
                if all(stored == utterance for stored, utterance in zip(stored_sentences, utterances)):
                    return stored_embeddings
                logger.info(f'Stored utterances do not match input utterances for cache key')

        logger.info(f'Sentence encoder cache miss for {len(utterances)} utterances')
        embeddings = self._sentence_embedding_model.encode(utterances)
        with open(cache_path, "wb") as fout:
            pickle.dump({'sentences': utterances, 'embeddings': embeddings}, fout, protocol=pickle.HIGHEST_PROTOCOL)

        return embeddings

"""
author: Jeiyoon
This work is based on the awesome work:
    https://github.com/amazon-research/dstc11-track2-intent-induction
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Set, Dict, List, Any, Optional

import numpy as np
from numpy import ndarray, argmax

from allennlp.common import Registrable

from sitod.data import DialogueDataset

@dataclass
class IntentClusteringContext:
    """
    Dialogue clustering context consisting of
        a list of dialogues and
        set of target turn IDs to be labeled with clusters.
    """
    dataset: DialogueDataset
    intent_turn_ids: Set[str]

    # output intermediate clustering results/metadata here
    outout_dir: Path = None

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
    Wrapper class for cluster labels
    """
    clusters: List[int]


@dataclass
class ClusteringContext:
    """
    Wrapper for ndarray containing clustering inputs
    """
    features: ndarray
    # output intermediate clustering results/metadata here
    output_dir: Path = None
    # dynamically inject parameters to clustering algorithm here
    parameters: Dict[str, Any] = field(default_factory=dict)


"""
-clustering algorithm 

OUR Intent Clustering MODEL HERE
OUR Intent Clustering MODEL HERE
OUR Intent Clustering MODEL HERE
OUR Intent Clustering MODEL HERE
OUR Intent Clustering MODEL HERE
"""


class ClusteringAlgorithm(Registrable):
    def cluster(self, context: ClusteringContext) -> ClusterData:
        """
        Predict cluster labels given a clustering context consisting of raw features and any parameters
        to dynamically pass to the clustering algorithm.
        :param context: clustering context
        :return: cluster labels
        """
        raise NotImplementedError


# https://www.sbert.net/
class SentenceEmbeddingModel(Registrable):
    def encode(self, utterances: List[str]) -> ndarray:
        """
        Encode a list of utterances as an array of real-valued vectors.
        :param utterances: original utterances
        :return: output encoding
        """
        raise NotImplementedError


"""
- sentence embedding

OUR Intent Clustering MODEL HERE
OUR Intent Clustering MODEL HERE
OUR Intent Clustering MODEL HERE
OUR Intent Clustering MODEL HERE
OUR Intent Clustering MODEL HERE
"""


@IntentClusteringModel.register('baseline_intent_clustering_model')
class BaselineIntentClusteringModel(IntentClusteringModel):
    def __init__(self,
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
                    # https://wikidocs.net/1015
                    labels.update(turn.intents)

        # compute sentence embeddings
        features = self._embedding_model.encode(utterances)
        # cluster sentence embeddings
        context = ClusteringContext(
            features,
            output_dir=context.outout_dir
        )
        result = self._clustering_algorithm.cluster(context)
        # map turn IDs to cluster labels
        return {turn_id: str(label) for turn_id, label in zip(turn_ids, result.clusters)}


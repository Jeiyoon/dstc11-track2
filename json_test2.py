# https://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_tsne.html
# https://pypi.org/project/umap-learn/
import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "6,7"

from setproctitle import *
setproctitle('k4ke')

import hashlib
import json
import jsonlines
import logging
import pickle
import numpy as np
from typing import List
from pathlib import Path
from tqdm import tqdm

from sklearn.manifold import TSNE
import umap
from matplotlib import pyplot as plt

from allennlp.common import Registrable
from sentence_transformers import SentenceTransformer



logger = logging.getLogger(__name__)

# https://bramhyun.tistory.com/44
# https://jimmy-ai.tistory.com/150
# trainset: /root/dstc11-track2-intent-induction/dstc11/development/dialogues.jsonl
# testset: /root/dstc11-track2-intent-induction/dstc11/development/test-utterances.jsonl

jsonl_path = '/data02/jeiyoon_park/dstc11-track2-intent-induction/dstc11/development/dialogues.jsonl'
# jsonl_path = '/data02/jeiyoon_park/dstc11-track2-intent-induction/dstc11/development/test-utterances.jsonl'

# print(path)
"""
sentence embedding models
"""
# https://www.sbert.net/
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
        encoder = SentenceTransformer(self._sentence_transformer)
        return encoder.encode(utterances)


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
        # test = self._sentence_embedding_model.encode('this is test.')
        # print(test)
        embeddings = self._sentence_embedding_model.encode(utterances)
        with open(cache_path, "wb") as fout:
            pickle.dump({'sentences': utterances, 'embeddings': embeddings}, fout, protocol=pickle.HIGHEST_PROTOCOL)

        return embeddings
"""
{'turn_id': 'insurance_947_020', 
'speaker_role': 'Customer', 
'utterance': "It's okay Alice.", 
'dialogue_acts': [], 
'intents': []}, 
"""

# model_name_or_path = 'sentence-transformers/average_word_embeddings_glove.840B.300d'
# model_name_or_path = 'sentence-transformers/all-mpnet-base-v2'
model_name_or_path = 'sentence-transformers/all-MiniLM-L12-v2'

transformer_model = SentenceTransformersModel(model_name_or_path)
# transformer_model = CachingSentenceEmbeddingModelSentenceTransformersModel(sentence_embedding_model=model_name_or_path,
#                                                                            cache_path=cache_path,
#                                                                            prefix=prefix)

utt_list = []
intent_list = []

gec_path = "/data02/jeiyoon_park/dstc11-track2-intent-induction/gec/Troy-1BW/new_1bw/train_source"
gec_list = []

# trainset: about 60k sentences
if __name__ == "__main__":
    with jsonlines.open(jsonl_path) as f:
        for idx, line in enumerate(f):
            # print("line {}".format(idx))

            for each_turn in line["turns"]:
                # print("each_turn: ", each_turn)
                utt_list.append(each_turn["utterance"])

    # gec
    with open(gec_path, 'r') as f:
        for line in tqdm(f):
            gec_list.append(line)

    # embeddings = transformer_model.encode(utt_list)
    embeddings = transformer_model.encode(gec_list)

    print("embedding done")

    # tsne = TSNE(n_components=2, random_state=0)
    X_2d = umap.UMAP().fit_transform(embeddings)
    # X_2d = tsne.fit_transform(embeddings)
    y = np.array([0] * X_2d.shape[0])

    plt.figure(figsize=(32, 32))

    colors = 'black'
    # for i, c, label in zip(target_ids, colors, target_names):
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, label=0)
    # plt.legend()
    plt.axis('auto')
    plt.show()







# if __name__ == "__main__":
#     with jsonlines.open(jsonl_path) as f:
#         for line in f:
#             utt_list.append(line['utterance'])
#             print("utt: ", line['utterance'])
#             intent_list.append(line['intent'])
#             print("intent: ", line['intent'])
#             # print(line["turns"][0])
#             print(" ")
#             # print(line["turns"][0]["utterance"])
#
#     print("separation done.")
#     """
#     - 913 utterances
#     - 22 intents:
#     {'ChangeSecurityQuestion': 29,
#     'RequestProofOfInsurance': 31,
#     'UpdateBillingFrequency': 29,
#     'CheckPaymentStatus': 31,
#     CancelAutomaticBilling': 31,
#     'FileClaim': 124,
#     'PayBill': 34,
#     'CreateAccount': 30,
#     'ChangeAddress': 31,
#     'AddDependent': 33,
#     'ReportBillingIssue': 32,
#     'GetPolicyNumber': 30,
#     'ChangePlan': 29,
#     'EnrollInPlan': 32,
#     'CancelPlan': 29,
#     'UpdatePaymentPreference': 29,
#     'ResetPassword': 29,
#     'RemoveDependent': 31,
#     'GetQuote': 181,
#     'CheckAccountBalance': 29,
#     'FindAgent': 29,
#     'ReportAutomobileAccident': 28}
#     """
#     # from collections import Counter
#     # test = Counter(intent_list)
#
#     # embeddings: ndarray(913, 300) / X
#     embeddings = transformer_model.encode(utt_list)
#
#     # intents: ndarray(913, )
#     intents = np.array(intent_list)
#
#     # tsne = TSNE(n_components=2, random_state=0)
#
#     # X_2d: ndarray(913, 2)
#     # X_2d = tsne.fit_transform(embeddings)
#     X_2d = umap.UMAP().fit_transform(embeddings)
#     y = intents
#
#
#     # Visualize the data
#     # target_ids: range(0, 22)
#     target_ids = range(len(list(set(intent_list))))
#
#     plt.figure(figsize=(16, 10))
#     target_names = list(set(intent_list))
#     # colors: tuple
#     colors = ('r',
#               'g',
#               'b',
#               'c',
#               'm',
#               'y',
#               'k',
#               'w',
#               'orange',
#               'purple', # 10
#               'navy',
#               'blueviolet',
#               'darkslategrey',
#               'cadetblue',
#               'violet',
#               'springgreen',
#               'slategray',
#               'aliceblue',
#               'cyan',
#               'royalblue', # 20
#               'indigo',
#               'saddlebrown')
#
#     for i, c, label in zip(target_ids, colors, target_names):
#         plt.scatter(X_2d[y == target_names[i], 0], X_2d[y == target_names[i], 1], c=c, label=label)
#     # plt.legend()
#     plt.show()
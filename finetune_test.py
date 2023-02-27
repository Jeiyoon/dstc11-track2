"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset
with MultipleNegativesRankingLoss. Entailnments are poisitive pairs and the contradiction on AllNLI dataset is added as a hard negative.
At every 10% training steps, the model is evaluated on the STS benchmark dataset
Usage:
python training_nli_v2.py
OR
python training_nli_v2.py pretrained_transformer_model_name
"""
from setproctitle import *
setproctitle('k4ke')

import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "6,7"

import jsonlines
import math
from sentence_transformers import models, losses, datasets
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


model_name = sys.argv[1] if len(sys.argv) > 1 else 'sentence-transformers/all-mpnet-base-v2'
train_batch_size = 128  # The larger you select this, the better the results (usually). But it requires more GPU memory
max_seq_length = 75
num_epochs = 1

# Save path of the model
model_save_path = 'test_output/training_test-model_v2_' + model_name.replace("/", "-") + '-' + datetime.now().strftime(
    "%Y-%m-%d_%H-%M-%S")

# Here we define our SentenceTransformer model
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Check if dataset exsist. If not, download and extract  it
# nli_dataset_path = 'data/AllNLI.tsv.gz'
train_dataset_path = '/data02/jeiyoon_park/dstc11-track2-intent-induction/dstc11/development/dialogues.jsonl'
dev_dataset_path = '/data02/jeiyoon_park/dstc11-track2-intent-induction/dstc11/development/test-utterances.jsonl'
# sts_dataset_path = 'data/stsbenchmark.tsv.gz'

# if not os.path.exists(nli_dataset_path):
#     util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)
#
# if not os.path.exists(sts_dataset_path):
#     util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)


# Read the AllNLI.tsv.gz file and create the training dataset
logging.info("Read AllNLI train dataset")

# def add_to_samples(sent1, sent2, label):
#     if sent1 not in train_data:
#         train_data[sent1] = {'contradiction': set(), 'entailment': set(), 'neutral': set()}
#     train_data[sent1][label].add(sent2)
def contrast_logits(self, embd1, embd2=None):
    feat1 = F.normalize(self.contrast_head(embd1), dim=1)
    if embd2 != None:
        feat2 = F.normalize(self.contrast_head(embd2), dim=1)
        return feat1, feat2
    else:
        return feat1

train_utt_list = []
dev_utt_list = []
intent_list = []

with jsonlines.open(train_dataset_path) as f:
    for idx, line in enumerate(f):
        # print("line {}".format(idx))

        for each_turn in line["turns"]:
            # print("each_turn: ", each_turn)
            train_utt_list.append(each_turn["utterance"])

# train_data = {}
# with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
#     reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
#     for row in reader:
#         if row['split'] == 'train':
#             sent1 = row['sentence1'].strip()
#             sent2 = row['sentence2'].strip()
#
#             add_to_samples(sent1, sent2, row['label'])
#             add_to_samples(sent2, sent1, row['label'])  # Also add the opposite

# train_samples = []
# for sent1, others in train_data.items():
#     if len(others['entailment']) > 0 and len(others['contradiction']) > 0:
#         train_samples.append(InputExample(
#             texts=[sent1, random.choice(list(others['entailment'])), random.choice(list(others['contradiction']))]))
#         train_samples.append(InputExample(
#             texts=[random.choice(list(others['entailment'])), sent1, random.choice(list(others['contradiction']))]))

# logging.info("Train samples: {}".format(len(train_samples)))
logging.info("Train samples: {}".format(len(train_utt_list)))

# Special data loader that avoid duplicates within a batch
train_dataloader = datasets.NoDuplicatesDataLoader(train_utt_list, batch_size=train_batch_size)

# Our training loss
# train_loss = losses.MultipleNegativesRankingLoss(model)
train_loss = losses.ContrastiveLoss(model)

# Read STSbenchmark dataset and use it as development set
# logging.info("Read STSbenchmark dev dataset")

# with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
#     reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
#     for row in reader:
#         if row['split'] == 'dev':
#             score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
#             dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
with jsonlines.open(dev_dataset_path) as f:
    for line in f:
        dev_utt_list.append(line['utterance'])

"""
여기 부터
"""
dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_utt_list, batch_size=train_batch_size,
                                                                 name='dev_evaluator')

# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=int(len(train_dataloader) * 0.1),
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          use_amp=False  # Set to True, if your GPU supports FP16 operations
          )

##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

test_samples = []
# with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
#     reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
#     for row in reader:
#         if row['split'] == 'test':
#             score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
#             test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_utt_list, batch_size=train_batch_size,
                                                                  name='test_dev_evaluator')
test_evaluator(model, output_path=model_save_path)
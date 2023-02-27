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

logger = logging.getLogger(__name__)

# trainset: /root/dstc11-track2-intent-induction/dstc11/development/dialogues.jsonl
# testset: /root/dstc11-track2-intent-induction/dstc11/development/test-utterances.jsonl

# jsonl_path = '/data02/jeiyoon_park/dstc11-track2-intent-induction/dstc11/development/dialogues.jsonl'
jsonl_path = '/data02/jeiyoon_park/dstc11-track2-intent-induction/dstc11/development/test-utterances.jsonl'

"""
{'turn_id': 'insurance_947_020', 
'speaker_role': 'Customer', 
'utterance': "It's okay Alice.", 
'dialogue_acts': [], 
'intents': []}, 
"""

utt_list = []
intent_list = []

"""
test
"""
if __name__ == "__main__":
    with jsonlines.open(jsonl_path) as f:
        for line in f:
            utterance = line['utterance']
            utterance_id = line['utterance_id']
            intent = line['intent']

            intent_list.append(intent)
            utt_list.append(utterance)

    print((list(set(intent_list))))
    print(len(list(set(intent_list))))
    print(len(utt_list))

"""
dev
"""
# if __name__ == "__main__":
#     with jsonlines.open(jsonl_path) as f:
#         for idx, line in enumerate(f):
#             print("idx: ", idx)
#             for turn in line['turns']:
#                 turn_id = turn['turn_id']
#                 speaker_role = turn['speaker_role']
#                 utterance = turn['utterance']
#                 dialogue_acts = turn['dialogue_acts']
#                 intents = turn['intents']
#                 print("{}: {} [{}, {}]".format(speaker_role, utterance, dialogue_acts, intents))
#                 utt_list.append(utterance)
#                 if intents != []:
#                     intent_list.extend(intents)
#             print(" ")
#             # if idx == 535: # 13, 29(multiple), 42, 183, 164, 596, 454
#             #     break
#     print("separation done.")
#     print(len(utt_list))
#     print(" ")
#     print(intent_list)
#     print(len(list(set(intent_list))))

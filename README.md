# Analysis of Utterance Embeddings and Clustering Methods Related to Intent Induction for Task-Oriented Dialogue

<img width="850" alt="Capture" src="https://user-images.githubusercontent.com/56618962/221478870-f5f66db1-415c-40ca-9219-951ac492177c.PNG">


- Paper: https://arxiv.org/abs/2212.02021
- This repository is basd on the awesome work: https://github.com/amazon-science/dstc11-track2-intent-induction

## Requirements

- Python 3.7
- PyTorch==1.11.0

```bash
# install dependencies
pip3 install -r requirements.txt
pip3 install setproctitle
```

## Dataset
- DSTC11 dataset

| **Dataset** |  **Domain**  |  **#Intents**  |  **#Utterances**  |
| :---: | :---: | :---: | :---: |
| **DSTC11-dev** |  insurance  |  22  |  66,875  |
| **DSTC11-test** |  insurance  |  22  |  913  |

- A sample segment of conversation transcript
```json
{"dialogue_id": "432", "turns": [{"turn_id": "insurance_432_000", "speaker_role": "Agent", "utterance": "Hello, you are currently speaking with Rivetown Insurance customer service. My name is Julian, How may I be of service to you?", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_001", "speaker_role": "Customer", "utterance": "The services I have being receiving from your company has being encouraging and my intent is to increase or enroll for more plans with you.", "dialogue_acts": ["InformIntent"], "intents": ["EnrollInPlan"]}, {"turn_id": "insurance_432_002", "speaker_role": "Agent", "utterance": "Whoa, that is such an encouraging word coming from you. So, which of the plans do you wish to register for?", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_003", "speaker_role": "Customer", "utterance": "I will like to enrol for the life insurance policy.", "dialogue_acts": ["InformIntent"], "intents": []}, {"turn_id": "insurance_432_004", "speaker_role": "Agent", "utterance": "That won't be an issue, I can help you get registered right away.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_005", "speaker_role": "Customer", "utterance": "Ok then, but first can I get my current policy number.", "dialogue_acts": ["InformIntent"], "intents": ["GetPolicyNumber"]}, {"turn_id": "insurance_432_006", "speaker_role": "Agent", "utterance": "Of course, I can help you get that.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_007", "speaker_role": "Customer", "utterance": "Please do!", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_008", "speaker_role": "Agent", "utterance": "Yap, I will need you to provide me with your first name and your last name.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_009", "speaker_role": "Customer", "utterance": "My first name is Bonnie.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_010", "speaker_role": "Agent", "utterance": "Bonnie is spelled B.O.N.I.E, correct?", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_011", "speaker_role": "Customer", "utterance": "Nah, thats wrong it has double N, as in B.O.N.N.I.E.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_012", "speaker_role": "Agent", "utterance": "Oh, sorry about that, please proceed with your last name.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_013", "speaker_role": "Customer", "utterance": "No offence taken, my last name is Wilson as in W.i.l.s.o.n.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_014", "speaker_role": "Agent", "utterance": "W.I.L.S.O.N.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_015", "speaker_role": "Customer", "utterance": "Yep, correct!", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_016", "speaker_role": "Agent", "utterance": "I will need you to provide me with your date of birth.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_017", "speaker_role": "Customer", "utterance": "My birthdate is second of may nineteen fifty six.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_018", "speaker_role": "Agent", "utterance": "Gotcha, please give me your customer number.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_019", "speaker_role": "Customer", "utterance": "That should be the number at the bottom of my card right?", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_020", "speaker_role": "Agent", "utterance": "Yep, correct!", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_021", "speaker_role": "Customer", "utterance": "Ok, let me get my card.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_022", "speaker_role": "Agent", "utterance": "Sure!", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_023", "speaker_role": "Customer", "utterance": "The number is one one one.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_024", "speaker_role": "Agent", "utterance": "One one one.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_025", "speaker_role": "Customer", "utterance": "Zero two one.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_026", "speaker_role": "Agent", "utterance": "Zero two one.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_027", "speaker_role": "Customer", "utterance": "Five two.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_028", "speaker_role": "Customer", "utterance": "Sorry that last number is four and not two.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_029", "speaker_role": "Agent", "utterance": "Ok, it's five four right?", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_030", "speaker_role": "Customer", "utterance": "That is correct.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_031", "speaker_role": "Agent", "utterance": "What is the current plan you are enrolled in?", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_032", "speaker_role": "Customer", "utterance": "I'm currently enrolled in Automobile Insurance.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_033", "speaker_role": "Agent", "utterance": "What plan sir?", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_034", "speaker_role": "Customer", "utterance": "I first did the basic plan for around one thousand dollars I think.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_035", "speaker_role": "Agent", "utterance": "Yep, correct!", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_036", "speaker_role": "Customer", "utterance": "Then I increased it to that of two thousand.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_037", "speaker_role": "Agent", "utterance": "You mean the Complete Auto.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_038", "speaker_role": "Customer", "utterance": "Exactly!", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_039", "speaker_role": "Agent", "utterance": "Gotcha, just hold on while I get your policy number sen to your mobile phone.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_040", "speaker_role": "Customer", "utterance": "Alright, take your time.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_041", "speaker_role": "Agent", "utterance": "Huh, your policy number has being sent to your phone. Please confirm reciept.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_042", "speaker_role": "Customer", "utterance": "Jeez, that was fast!", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_043", "speaker_role": "Agent", "utterance": "That is why we are the best at giving our customers utmost satisfaction.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_044", "speaker_role": "Agent", "utterance": "speaking about the plan you wanted to enrol for. Can you tell me which particular one it is you want to enrol for?", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_045", "speaker_role": "Customer", "utterance": "Mhm, about that I will like to get a quote for the Life insurance first.", "dialogue_acts": ["InformIntent"], "intents": ["GetQuote"]}, {"turn_id": "insurance_432_046", "speaker_role": "Agent", "utterance": "Ok the, that won be an issue. We have the.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_047", "speaker_role": "Customer", "utterance": "Can i get that sent to my mail or can I get the information on your website. It will afford me enough time to read understand it well. You know what the say about old age ?", "dialogue_acts": ["InformIntent"], "intents": []}, {"turn_id": "insurance_432_048", "speaker_role": "Agent", "utterance": "[Laughter] No problem, please provide me with your email address.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_049", "speaker_role": "Customer", "utterance": "Bonnie W Wilson at dftmail dot com.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_050", "speaker_role": "Agent", "utterance": "Gotcha, just a second.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_051", "speaker_role": "Agent", "utterance": "The mail has being sent.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_052", "speaker_role": "Customer", "utterance": "Thanks, for the past few decades of my existence, this is actually one of the best customer service I have ever heard, you are sure the best the company has.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_053", "speaker_role": "Agent", "utterance": "We are all professionals sir, that is why we are the best around.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_054", "speaker_role": "Customer", "utterance": "Whoa! I see.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_055", "speaker_role": "Agent", "utterance": "Is there any other thing I can help you with?", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_056", "speaker_role": "Customer", "utterance": "Nah, that will be all for now. I will contact back when I'm done with the mail.", "dialogue_acts": [], "intents": []}, {"turn_id": "insurance_432_057", "speaker_role": "Agent", "utterance": "That will be great, thank you for chosing Rivertown Insurance. Bye!", "dialogue_acts": [], "intents": []}]}
```

## Run: Task 1 and Task 2

### Task 1: Intent Clustering

```bash
# run intent clustering (Task 1) baselines and evaluation
python3 -m sitod.run_experiment \
--data_root_dir dstc11 \
--experiment_root_dir results \
--config configs/run-intent-clustering-baselines.jsonnet
```

- Best Performance (MiniLM-L12 & K-means)

| **Model** |  **NMI**  |  **ARI**  |  **ACC**  |  **Precision**  |  **Recall**  |  **F1**  |  **Example Coverage**  |  **K**  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ours** |  63.1  |  38.9  |  54.9  |  68.0  |  54.9  |  60.8  |  100.0  |  31  |

- Experimental Result (Clustering algorithm: K-means)

| **Model** |  **NMI**  |  **ARI**  |  **ACC**  |  **Precision**  |  **Recall**  |  **F1**  |  **Example Coverage**  |  **K**  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EASE-RoBERTa-Multilingual**  |  33.4  |  14.2  |  28.0  |  28.0  |  69.0  |  39.9  |  43.9  |  5  |
| **EASE-BERT** |  36.1  |  18.1  |  35.3  |  36.8  |  58.9  |  45.3  |  53.5  |  8  |
| **EASE-BERT-Multilingual** |  38.4  |  10.9  |  23.6  |  42.7  |  24.2  |  30.9  |  87.6  |  44  |
| **EASE-RoBERTa** |  45.8  |  25.4  |  40.0  |  43.7  |  60.7  |  50.9  |  65.6  |  12  |
| **DiffCSE-BERT-Trans** |  43.8  |  15.5  |  29.4  |  49.5  |  30.6  |  37.8  |  90.1  |  40  |
| **DiffCSE-BERT-Sts** |  46.6  |  16.9  |  29.9  |  50.4  |  30.8  |  38.2  |  89.9  |  43  |
| **DiffCSE-RoBERTa-Sts** |  53.9  |  29.0  |  45.1  |  57.2  |  46.8  |  51.5  |  91.1  |  30  |
| **DiffCSE-RoBERTa-Trans** |  55.0  |  23.1  |  35.7  |  60.6  |  35.7  |  44.9  |  96.7  |  50  |
| **SimCSE-BERT-Unsupervised** |  31.7  |  13.3  |  27.9  |  27.9  |  65.0  |  39.0  |  43.9  |  5  |
| **SimCSE-BERT-Large-Unsupervised** |  47.0  |  25.7  |  38.4  |  46.9  |  46.9  |  46.9  |  76.8  |  19  |
| **SimCSE-RoBERTa-Unsupervised** |  51.2  |  29.0  |  44.4  |  49.7  |  61.6  |  55.0  |  70.5  |  14  |
| **SimCSE-BERT-Large-Supervised** |  53.0  |  27.8  |  39.8  |  55.0  |  42.2  |  47.8  |  91.0  |  31  |
| **SimCSE-BERT-Supervised** |  53.1  |  24.4  |  39.0  |  60.5  |  39.5  |  47.8  |  96.3  |  44  |
| **SimCSE-RoBERTa-Large-Unsupervised** |  53.2  |  25.9  |  42.2  |  56.8  |  43.8  |  49.5  |  91.7  |  32  |
| **SimCSE-RoBERTa-Supervised** |  56.6  |  28.8  |  42.7  |  60.8  |  43.3  |  50.6  |  91.3  |  36  |
| **SimCSE-RoBERTa-Large-Supervised** |  56.8  |  28.9  |  41.3  |  62.5  |  41.6  |  49.9  |  98.4  |  42  |
| **Glove-Avg** |  30.5  |  7.0  |  20.6  |  34.6  |  22.2  |  27.0  |  92.2  |  50  |
| **MPNet** |  59.3  |  32.3  |  46.1  |  66.0  |  47.1  |  54.9  |  96.5  |  42  |
| **MiniLM-L6** | 59.3  |  35.7  |  52.6  |  62.2  |  54.9  |  58.4  |  92.4  |  28  |
| **MiniLM-MultiQA** |  61.7  |  38.2  |  55.1  |  66.6  |  55.4  |  60.5  |  98.8  |  30  |
| **MiniLM-L12** |  63.1  |  38.9  |  54.9  |  68.0  |  54.9  |  60.8  |  100.0  |  31  |

### Task 2: Intent Induction

```bash
# run open intent induction (Task 2) baselines and evaluation
python3 -m sitod.run_experiment \
--data_root_dir dstc11 \
--experiment_root_dir results \
--config configs/run-open-intent-induction-baselines.jsonnet
```

- Best Performance (MiniLM-MultiQA & Agglomerative)

| **Model** |  **NMI**  |  **ARI**  |  **ACC**  |  **Precision**  |  **Recall**  |  **F1**  |  **Example Coverage**  |  **K**  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ours** |  81.0  |  64.2  |  66.5  |  75.5  |  80.6  |  78.0  |  86.2  |  25  |

- Experimental Result (Clustering algorithm: K-means)

| **Model** |  **NMI**  |  **ARI**  |  **ACC**  |  **Precision**  |  **Recall**  |  **F1**  |  **Example Coverage**  |  **K**  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EASE-BERT-Multilingual** |  27.0  |  12.3  |  23.9  |  24.1  |  72.9  |  36.2  |  26.4  |  5  |
| **EASE-BERT** |  53.1  |  25.6  |  42.8  |  50.2  |  57.4  |  53.5  |  80.0  |  26  |
| **EASE-RoBERTa-Multilingual** |  59.6  |  41.8  |  53.7  |  57.8  |  68.8  |  62.8  |  86.0  |  34  |
| **EASE-RoBERTa** |  60.5  |  50.7  |  52.7  |  55.8  |  70.0  |  62.1  |  83.1  |  28  |
| **DiffCSE-BERT-Trans** |  21.0  |  11.5  |  23.3  |  23.7  |  77.5  |  36.3  |  35.5  |  6  |
| **DiffCSE-BERT-Sts** |  49.6  |  22.3  |  40.2  |  46.1  |  58.4  |  51.5  |  93.2  |  31  |
| **DiffCSE-RoBERTa-Sts** |  65.4  |  42.3  |  53.7  |  59.4  |  65.9  |  62.5  |  89.8  |  31  |
| **DiffCSE-RoBERTa-Trans** |  65.4  |  53.3  |  57.6  |  63.1  |  71.1  |  66.8  |  90.0  |  28  |
| **SimCSE-BERT-Large-Unsupervised** |  33.1  |  19.1  |  28.0  |  28.0  |  77.5  |  41.2  |  33.4  |  5  |
| **SimCSE-BERT-Unsupervised** |  58.3  |  29.6  |  46.4  |  58.5  |  59.1  |  58.8  |  83.1  |  32  |
| **SimCSE-BERT-Large-Supervised** |  59.4  |  29.1  |  47.0  |  54.2  |  60.5  |  57.2  |  86.2  |  30  |
| **SimCSE-RoBERTa-Supervised** |  61.4  |  34.3  |  47.1  |  54.3  |  62.0  |  57.9  |  80.3  |  29  |
| **SimCSE-BERT-Supervised** |  62.7  |  37.3  |  52.6  |  63.4  |  59.8  |  61.6  |  89.6  |  34  |
| **SimCSE-RoBERTa-Large-Unsupervised** |  67.8  |  51.1  |  57.7  |  64.2  |  70.2  |  67.1  |  86.4  |  32  |
| **SimCSE-RoBERTa-Unsupervised** |  68.6  |  48.0  |  54.7  |  65.6  |  71.2  |  68.3  |  86.1  |  31  |
| **SimCSE-RoBERTa-Large-Supervised** |  69.2  |  38.7  |  52.1  |  62.4  |  70.0  |  66.0  |  93.8  |  32  |
| **Glove-Avg** |  35.0  |  18.8  |  29.1  |  36.3  |  51.2  |  42.4  |  77.0  |  29  |
| **MPNet** |  72.8  |  41.6  |  60.2  |  65.6  |  76.3  |  70.6  |  86.5  |  26  |
| **MiniLM-L6** | 73.2  |  47.1  |  57.1  |  66.0  |  72.2  |  69.0  |  83.1  |  25  |
| **MiniLM-MultiQA** |  74.7  |  52.8  |  61.2  |  70.5  |  74.9  |  72.7  |  83.4  |  25  |
| **MiniLM-L12** |  77.4  |  54.5  |  63.2  |  70.5  |  79.1  |  74.6  |  80.0  |  24  |

## Inference

~~~
Yeah where do I go on here to pay my bill?	34
Hi, I need to speak to someone about my bill.	34
Okay. When will I see that in my bill?	34
Would it be on like any of my other bills?	34
Okay. Where at on the bill?	34
Oh okay. Where at on the bill. I'm looking now.	34
Thank you. How much will this make my bill go up?	34
Well I should probably pay my bill while I have you on the line.	34
How can I make payments going forward without calling in?	34
Hi. I need to check my bill? And maybe pay my bill?	34
Oh, man. Is there any other way I can check my bill and pay it?	34
So, my bill after this one won't be til-.	34
And when is that bill due every month?	34
September first alright. And I will get a bill in the mail for that?	34
Hi I am calling because I received a wrong bill.	34
And from here I can pay my bill online right?	34
Hi I need to change my billing address on my policy.	34
~~~

## Citation

- Please consider citing our paper if it is of help to you ☺

```bib
@misc{https://doi.org/10.48550/arxiv.2212.02021,
  doi = {10.48550/ARXIV.2212.02021},
  url = {https://arxiv.org/abs/2212.02021},
  author = {Jeiyoon Park, Yoonna Jang, Chanhee Lee and Heuiseok Lim},
  title = {Analysis of Utterance Embeddings and Clustering Methods Related to Intent Induction for Task-Oriented Dialogue},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```




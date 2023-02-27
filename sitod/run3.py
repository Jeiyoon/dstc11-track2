import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

# Import our pretrained model.
tokenizer = AutoTokenizer.from_pretrained("sosuke/ease-bert-base-multilingual-cased")
model = AutoModel.from_pretrained("sosuke/ease-bert-base-multilingual-cased")

# Set pooler.
pooler = lambda last_hidden, att_mask: (last_hidden * att_mask.unsqueeze(-1)).sum(1) / att_mask.sum(-1).unsqueeze(-1)

# Tokenize input texts.
texts = [
    "Ils se préparent pour un spectacle à l'école.",
    "They are preparing for a show at school.",
    "Two medical professionals in green look on at something."
]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Get the embeddings
with torch.no_grad():
    last_hidden = model(**inputs, output_hidden_states=True, return_dict=True).last_hidden_state
embeddings = pooler(last_hidden, inputs["attention_mask"])

# Calculate cosine similarities
cosine_sim_0_1 = 1 - cosine(embeddings[0], embeddings[1])
cosine_sim_0_2 = 1 - cosine(embeddings[0], embeddings[2])

print(f"Cosine similarity between {texts[0]} and {texts[1]} is {cosine_sim_0_1}")
print(f"Cosine similarity between {texts[0]} and {texts[2]} is {cosine_sim_0_2}")
import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
MODEL_NAME = 'allenai/specter2_base'

df = pd.read_parquet(INPUT_PARQUET)
#print(df.head())

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

df['title'] = df['title'].fillna("")
df['abstract'] = df['abstract'].fillna("")

df['text'] = df['title'] + tokenizer.sep_token + df['abstract']

#print(df.head())

model = SentenceTransformer(MODEL_NAME)
texts = df['text'].tolist()

embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

total_texts = len(texts)
embedding_dim = embeddings.shape[1] 
first_emb_norm = np.linalg.norm(embeddings[0]) 

print("\n--- Результати ---")
print(f"Загальна кількість оброблених текстів: {total_texts}")
print(f"Розмірність ембеддингів: {embedding_dim}")
print(f"Норма першого ембеддингу: {first_emb_norm:.4f}")

os.makedirs('embeddings', exist_ok=True)

save_path = INPUT_EMBEDDINGS
np.save(save_path, embeddings)
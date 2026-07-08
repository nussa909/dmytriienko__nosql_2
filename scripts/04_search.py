import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "arxiv-papers"
INPUT_PARQUET = "data/arxiv_subset.parquet"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet(INPUT_PARQUET)  # для отримання повного abstract

query = "teaching machines to recognize objects in pictures"
query_vector = model.encode(query).tolist()

results = index.query(
    vector=query_vector,
    top_k=TOP_K,
    include_metadata=True
)

print(f"Запит: '{query}':")
print("=" * 60)
    
for i, match in enumerate(results.get('matches', []), 1):
    score = match.get('score', 0.0)
    metadata = match.get('metadata', {})
    
    # Безпечно витягуємо метадані
    title = metadata.get('title', 'Без назви')
    category = metadata.get('category', 'Не вказано')
    year = metadata.get('year', 'Невідомо')
    abstract = metadata.get('abstract', 'Анотація відсутня')
    
    print(f"{i}. [Схожість: {score:.4f}] {title.strip()}")
    print(f"   Категорія: {category} | Рік публікації: {year}")
    print(f"   Анотація (уривок): {abstract.strip()[:250]}...")
    print("-" * 60)
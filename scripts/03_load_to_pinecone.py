import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="dotproduct",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"  # обирайте регіон ближче до ваших користувачів
        ),
    )

index = pc.Index(INDEX_NAME)

df = pd.read_parquet(INPUT_PARQUET)
embeddings = np.load(INPUT_EMBEDDINGS)

if len(df) != len(embeddings):
        raise ValueError(f"Помилка: {len(df)} текстів у датасеті, але {len(embeddings)} векторів.")

total_records = len(df)

print(df.head())

for i in tqdm(range(0, total_records, BATCH_SIZE), desc="Завантаження батчів"):
        batch_df = df.iloc[i : i + BATCH_SIZE]
        batch_embeddings = embeddings[i : i + BATCH_SIZE]
        
        vectors_to_upsert = []
        
        for j, (_, row) in enumerate(batch_df.iterrows()):
            global_index = i + j
            vector_id = f"paper_{global_index}"
            
            year_val = row.get('year')
            year = int(year_val) if pd.notna(year_val) else 0

            metadata = {
                "arxiv_id": str(row.get('id', '')), 
                "title": str(row.get('title', '')),
                "abstract": str(row.get('abstract', ''))[:500],
                "authors": str(row.get('authors', ''))[:200],
                "year": year,
                "category": str(row.get('category', ''))
            }
            
            vectors_to_upsert.append({
                "id": vector_id,
                "values": batch_embeddings[j].tolist(),
                "metadata": metadata
            })
            
        index.upsert(vectors=vectors_to_upsert)

stats = index.describe_index_stats()
total_vectors_in_index = stats.total_vector_count
    
print("*" * 40)
print(f"Загальна кількість векторів в індексі: {total_vectors_in_index}")
print("*" * 40)

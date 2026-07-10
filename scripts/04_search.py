import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "arxiv-papers"
INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet(INPUT_PARQUET)  # для отримання повного abstract
embeddings = np.load(INPUT_EMBEDDINGS)

def encode_query(query: str) -> list:
    embeddings = model.encode(query, normalize_embeddings=True)
    return embeddings.tolist()

def perform_semantic_search(query_text: str, filter_dict: dict = None ):
    query_vector = encode_query(query_text) 
    results = index.query(
        vector=query_vector,
        top_k=TOP_K,
        include_metadata=True,
        filter = filter_dict
    )

    matches = results.get('matches', [])
    if not matches:
        print(" За цими фільтрами нічого не знайдено.")
        return
    
    print(f"Запит: '{query_text}':")
    print("=" * 60)

    for i, match in enumerate(matches, 1):
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


query1 = "teaching machines to recognize objects in pictures"
results = perform_semantic_search(query1)


query2 = "reinforcement learning"
results = perform_semantic_search(query2, filter_dict = {
                                            "$and": [
                                                {"category": {"$eq": "cs.LG"} },
                                                {"year": {"$gte": 2021}}
                                            ]
                                        })

results = perform_semantic_search(query2, filter_dict = {"year": {"$lte": 2015}})

def compare_local_metrics(query_text: str):
    if df is None or embeddings is None:
        return

    q = encode_query(query_text)
    E = embeddings
    
    norms_E = np.linalg.norm(E, axis=1)
    norm_q = np.linalg.norm(q)
    
    cosine_scores = np.dot(E, q) / (norms_E * norm_q)
    top_cosine = np.argsort(cosine_scores)[::-1][:TOP_K]
    
    dot_scores = np.dot(E, q)
    top_dot = np.argsort(dot_scores)[::-1][:TOP_K]
    
    l2_scores = np.linalg.norm(E - q, axis=1)
    top_l2 = np.argsort(l2_scores)[:TOP_K] 

    metrics = {
        "1. Cosine Similarity": (top_cosine, cosine_scores),
        "2. Dot Product": (top_dot, dot_scores),
        "3. L2 Distance (Менше - краще)": (top_l2, l2_scores)
    }

    for metric_name, (indices, scores) in metrics.items():
        print(f"Метрика: {metric_name}")
        for rank, idx in enumerate(indices, 1):
            title = str(df.iloc[idx].get('title', 'Без назви')).strip()
            score = scores[idx]
            print(f"{rank}. [Score/Dist: {score:.4f}] {title}")

compare_local_metrics(query1)

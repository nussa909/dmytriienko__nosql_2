import os
import math
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 10   # беремо ширше, щоб RRF міг переранжувати

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)

df['title'] = df['title'].fillna('')
df['abstract'] = df['abstract'].fillna('')
articles_db = df.set_index('id').to_dict('index')

corpus = (df['title'] + " " + df['abstract']).tolist()
tokenized_corpus = [doc.lower().split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

arxiv_ids = df['id'].tolist()

def search_bm25(query: str):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    
    top_indices = scores.argsort()[::-1][:TOP_K]
    results = [(arxiv_ids[i], scores[i]) for i in top_indices if scores[i] > 0]
    return results

def search_vector(query: str):
    query_vector = model.encode(query, normalize_embeddings=True).tolist()
    res = index.query(vector=query_vector, top_k=TOP_K, include_metadata=True)
    
    results = []
    for match in res.get('matches', []):
        a_id = match['metadata'].get('arxiv_id')
        if a_id:
            results.append((str(a_id), match['score']))
    return results

def search_hybrid_rrf(query: str, k_constant: int = 60):
    bm25_results = search_bm25(query)
    vector_results = search_vector(query)
    
    rrf_scores = {}
    
    for rank, (doc_id, _) in enumerate(bm25_results, 1):
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = 0.0
        rrf_scores[doc_id] += 1.0 / (k_constant + rank)
        
    for rank, (doc_id, _) in enumerate(vector_results, 1):
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = 0.0
        rrf_scores[doc_id] += 1.0 / (k_constant + rank)
        
    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_rrf[:TOP_K]

def print_results(results, title_text, score_type="Score"):
    print(f"--- {title_text} ---")
    if not results:
        print("Нічого не знайдено.")
        return
        
    for i, (doc_id, score) in enumerate(results, 1):
        article = articles_db.get(doc_id, {})
        title = article.get('title', 'Невідомий заголовок')
        print(f"{i}. [{score_type}: {score:.4f}] {title} (ID: {doc_id})")

def run_comparison(query: str):
    print(f"\n\n{'-'*80}")
    print(f"Запит: '{query}'")
    print(f"{'-'*80}")
    
    bm25_top5 = search_bm25(query)
    print_results(bm25_top5, "Топ-5 BM25 (Лексичний пошук)", "BM25 Score")

    vector_top5 = search_vector(query)
    print_results(vector_top5, "Топ-5 Pinecone (Векторний пошук)", "Cosine/Dot")
    
    hybrid_top5 = search_hybrid_rrf(query)
    print_results(hybrid_top5, "Топ-5 HYBRID (RRF: BM25 + Vector)", "RRF Score")

test_queries = [
    "BERT fine-tuning",                                     
    "Yann LeCun convolutional networks",                    
    "making computers understand human emotions from text"  
]

for q in test_queries:
    run_comparison(q)

run_comparison("reinforcement learning")
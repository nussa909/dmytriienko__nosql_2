import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

MODEL_NAME = "allenai/specter2_base"
INDEX_FIXED = "arxiv-chunks-fixed"
INDEX_SEMANTIC = "arxiv-chunks-semantic"
VECTOR_DIM = 768

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")

df['abstract_len'] = df['abstract'].fillna("").apply(len)
top_30_df = df.nlargest(30, 'abstract_len').copy()

def fixed_size_chunking(text, chunk_size=200, overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=lambda x: len(x.split()),
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_text(text.strip())
    return chunks


def semantic_chunking(
    text: str,
    model: SentenceTransformer,
    threshold: float = 0.7,
    min_chunk_size: int = 50,
):
    """
    Ділить текст на семантично зв'язні блоки.
    Новий chunk починається, коли косинусна схожість
    між сусідніми реченнями падає нижче threshold.
    """
    # Просте розділення на речення
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    if len(sentences) < 2:
        return sentences

    # Отримуємо ембеддинги речень
    embeddings = model.encode(sentences, normalize_embeddings=True)

    # Косинусна схожість між сусідніми реченнями
    similarities = [
        float(np.dot(embeddings[i], embeddings[i + 1]))
        for i in range(len(embeddings) - 1)
    ]

    chunks, current_chunk = [], [sentences[0]]
    for i, sim in enumerate(similarities):
        if sim < threshold and len(" ".join(current_chunk)) >= min_chunk_size:
            chunks.append(". ".join(current_chunk) + ".")
            current_chunk = [sentences[i + 1]]
        else:
            current_chunk.append(sentences[i + 1])

    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")

    return chunks

def prepare_pinecone_index(index_name):
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="dotproduct", 
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    else:
        print(f"Індекс '{index_name}' вже існує.")
    return pc.Index(index_name)


def process_and_upsert_chunks(index, chunks_data, batch_size=64):
    texts_to_encode = [item['text'] for item in chunks_data]
    
    embeddings = model.encode(
        texts_to_encode, 
        batch_size=batch_size, 
        show_progress_bar=True, 
        normalize_embeddings=True
    )
    
    vectors_to_upsert = []
    
    for i in range(len(chunks_data)):
        data = chunks_data[i]
        vector = {
            "id": data["id"],
            "values": embeddings[i].tolist(),
            "metadata": data["metadata"]
        }
        vectors_to_upsert.append(vector)
        
    for i in tqdm(range(0, len(vectors_to_upsert), batch_size)):
        batch = vectors_to_upsert[i : i + batch_size]
        index.upsert(vectors=batch)
        
    stats = index.describe_index_stats()
    print(f"Завантаження завершено. Векторів в індексі: {stats.total_vector_count}")

def search_chunks(query, idx_fixed, idx_semantic, top_k=5):
    print(f"{'-'*80}")
    print(f"Запит: '{query}'")
    print(f"{'-'*80}")
    
    query_vector = model.encode(query, normalize_embeddings=True).tolist()

    def print_results(index_obj, name):
        print(f"--- Результати з індексу: {name} ---")
        res = index_obj.query(vector=query_vector, top_k=top_k, include_metadata=True)
        
        for i, match in enumerate(res.get('matches', []), 1):
            meta = match['metadata']
            score = match['score']
            print(f"{i}. [Схожість: {score:.4f}] {meta['title']}")
            print(f"Чанк #{meta['chunk_number']} (Рік: {meta['year']}, Категорія: {meta['category']})")
            print(f"Текст: \"{meta['chunk_text']}\"")
            print("-" * 60)

    print_results(idx_fixed, "FIXED SIZE CHUNKING")
    print_results(idx_semantic, "SEMANTIC CHUNKING")


fixed_chunks_data = []
semantic_chunks_data = []

for _, row in top_30_df.iterrows():
    arxiv_id = str(row.get('id', 'unknown'))
    title = str(row.get('title', 'Без назви'))
    abstract = str(row.get('abstract', ''))
    
    year_val = row.get('year')
    year = int(year_val) if pd.notna(year_val) else 0
    category = str(row.get('category', ''))

    f_chunks = fixed_size_chunking(abstract, chunk_size=200, overlap=50)
    s_chunks = semantic_chunking(abstract, model, threshold=0.6)

    for idx, text in enumerate(f_chunks):
            fixed_chunks_data.append({
                "id": f"{arxiv_id}_fixed_{idx}",
                "text": text,
                "metadata": {
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "chunk_text": text,
                    "chunk_number": idx,
                    "year": year,
                    "category": category
                }
            })
            
    for idx, text in enumerate(s_chunks):
        semantic_chunks_data.append({
            "id": f"{arxiv_id}_semantic_{idx}",
            "text": text,
            "metadata": {
                "arxiv_id": arxiv_id,
                "title": title,
                "chunk_text": text,
                "chunk_number": idx,
                "year": year,
                "category": category
            }
        })


idx_fixed = prepare_pinecone_index(INDEX_FIXED)
idx_semantic = prepare_pinecone_index(INDEX_SEMANTIC)

process_and_upsert_chunks(idx_fixed, fixed_chunks_data)
process_and_upsert_chunks(idx_semantic, semantic_chunks_data)


test_queries = [
    "quantum entanglement in complex systems",
    "deep learning architectures for natural language",
    "how to mitigate risks in autonomous driving"
]

for query in test_queries:
    search_chunks(query, idx_fixed, idx_semantic)
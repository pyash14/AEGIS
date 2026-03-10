import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config.settings import NUTRITION_INDEX_DIR, MEDICAL_INDEX_DIR, TOP_K_CHUNKS

model = SentenceTransformer('all-MiniLM-L6-v2')

# Nutrition index
_nutrition_index = None
_nutrition_metadata = None

# Medical index
_medical_index = None
_medical_chunks = None

def _load_nutrition_index():
    global _nutrition_index, _nutrition_metadata
    if _nutrition_index is None:
        _nutrition_index = faiss.read_index(os.path.join(NUTRITION_INDEX_DIR, 'index.faiss'))
        with open(os.path.join(NUTRITION_INDEX_DIR, 'metadata.json'), 'r') as f:
            _nutrition_metadata = json.load(f)

def _load_medical_index():
    global _medical_index, _medical_chunks
    if _medical_index is None:
        _medical_index = faiss.read_index(os.path.join(MEDICAL_INDEX_DIR, 'index.faiss'))
        with open(os.path.join(MEDICAL_INDEX_DIR, 'chunks.json'), 'r') as f:
            _medical_chunks = json.load(f)

def retrieve_nutrition(query: str, top_k: int = 3) -> list:
    try:
        _load_nutrition_index()
        embedding = model.encode([query])
        embedding = np.array(embedding).astype('float32')
        distances, indices = _nutrition_index.search(embedding, top_k)
        results = []
        for idx in indices[0]:
            if idx < len(_nutrition_metadata):
                item = _nutrition_metadata[idx].copy()
                item['per_100g'] = True
                results.append(item)
        return results
    except Exception as e:
        print(f'Nutrition retrieval error: {str(e)}')
        return []

def retrieve_medical(query: str, top_k: int = 4) -> list:
    try:
        _load_medical_index()
        embedding = model.encode([query])
        embedding = np.array(embedding).astype('float32')
        distances, indices = _medical_index.search(embedding, top_k)
        results = []
        for idx in indices[0]:
            if idx < len(_medical_chunks):
                results.append(_medical_chunks[idx])
        return results
    except Exception as e:
        print(f'Medical retrieval error: {str(e)}')
        return []

if __name__ == '__main__':
    print('Testing nutrition retriever...')
    results = retrieve_nutrition('oatmeal')
    for r in results:
        print(r)

    print('\nTesting medical retriever...')
    results = retrieve_medical('what is IOB insulin on board')
    for r in results:
        print(r[:200])
        print('---')
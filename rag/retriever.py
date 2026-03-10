import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config.settings import NUTRITION_INDEX_DIR, TOP_K_CHUNKS

model = SentenceTransformer('all-MiniLM-L6-v2')

_nutrition_index = None
_nutrition_metadata = None

def _load_nutrition_index():
    global _nutrition_index, _nutrition_metadata
    if _nutrition_index is None:
        _nutrition_index = faiss.read_index(os.path.join(NUTRITION_INDEX_DIR, 'index.faiss'))
        with open(os.path.join(NUTRITION_INDEX_DIR, 'metadata.json'), 'r') as f:
            _nutrition_metadata = json.load(f)

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
        print(f'Retrieval error: {str(e)}')
        return []

if __name__ == '__main__':
    print('Testing nutrition retriever...')
    results = retrieve_nutrition('oatmeal')
    for r in results:
        print(r)
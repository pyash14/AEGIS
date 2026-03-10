import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from config.settings import NUTRITION_INDEX_DIR, FDC_CSV_PATH

model = SentenceTransformer('all-MiniLM-L6-v2')

def build_nutrition_index():
    try:
        print('Loading FDC CSV...')
        df = pd.read_csv(FDC_CSV_PATH)
        print(f'Loaded {len(df)} foods')

        food_names = df['food_name'].tolist()
        metadata = df[['food_name', 'carbs_g', 'protein_g', 'fat_g', 'calories']].to_dict(orient='records')

        print('Building embeddings (this takes 1-2 mins)...')
        embeddings = model.encode(food_names, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')

        print('Building FAISS index...')
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        os.makedirs(NUTRITION_INDEX_DIR, exist_ok=True)
        faiss.write_index(index, os.path.join(NUTRITION_INDEX_DIR, 'index.faiss'))

        with open(os.path.join(NUTRITION_INDEX_DIR, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

        print(f'Nutrition index built — {len(food_names)} foods indexed')

    except Exception as e:
        print(f'Error building nutrition index: {str(e)}')

if __name__ == '__main__':
    build_nutrition_index()
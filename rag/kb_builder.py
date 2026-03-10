import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import faiss
import numpy as np
import pandas as pd
import fitz  # pymupdf
from sentence_transformers import SentenceTransformer
from config.settings import (
    NUTRITION_INDEX_DIR, FDC_CSV_PATH,
    MEDICAL_INDEX_DIR, MEDICAL_DOCS_DIR
)

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ''
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except Exception as e:
        print(f'  Error reading {pdf_path}: {str(e)}')
        return ''

def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i+chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def build_medical_index():
    try:
        print('Loading medical KB documents...')
        all_chunks = []

        files = os.listdir(MEDICAL_DOCS_DIR)
        pdf_files = [f for f in files if f.endswith('.pdf')]
        txt_files = [f for f in files if f.endswith('.txt')]

        print(f'Found {len(pdf_files)} PDFs and {len(txt_files)} TXT files')

        # Process PDFs
        for fname in sorted(pdf_files):
            fpath = os.path.join(MEDICAL_DOCS_DIR, fname)
            text = extract_text_from_pdf(fpath)
            if text:
                chunks = chunk_text(text)
                all_chunks.extend(chunks)
                print(f'  {fname} -> {len(chunks)} chunks')
            else:
                print(f'  {fname} -> EMPTY, skipped')

        # Process TXT files
        for fname in sorted(txt_files):
            fpath = os.path.join(MEDICAL_DOCS_DIR, fname)
            with open(fpath, 'r') as f:
                text = f.read()
            if text.strip():
                chunks = chunk_text(text)
                all_chunks.extend(chunks)
                print(f'  {fname} -> {len(chunks)} chunks')

        if not all_chunks:
            print('No content found in medical docs folder!')
            return

        print(f'\nTotal chunks: {len(all_chunks)}')
        print('Building medical embeddings...')
        embeddings = model.encode(all_chunks, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        os.makedirs(MEDICAL_INDEX_DIR, exist_ok=True)
        faiss.write_index(index, os.path.join(MEDICAL_INDEX_DIR, 'index.faiss'))

        with open(os.path.join(MEDICAL_INDEX_DIR, 'chunks.json'), 'w') as f:
            json.dump(all_chunks, f)

        print(f'Medical index built — {len(all_chunks)} chunks indexed')

    except Exception as e:
        print(f'Error building medical index: {str(e)}')

def build_nutrition_index():
    try:
        print('Loading FDC CSV...')
        df = pd.read_csv(FDC_CSV_PATH)
        print(f'Loaded {len(df)} foods')

        food_names = df['food_name'].tolist()
        metadata = df[['food_name', 'carbs_g', 'protein_g', 'fat_g', 'calories']].to_dict(orient='records')

        print('Building nutrition embeddings...')
        embeddings = model.encode(food_names, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')

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
    build_medical_index()
    print()
    build_nutrition_index()
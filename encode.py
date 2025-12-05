#!/usr/bin/env python3
"""
encode.py
- Loads BAAI/bge-base-en-v1.5
- Provides encode_query(text) -> np.ndarray of shape (768,)
"""

from functools import lru_cache
from typing import Union
import numpy as np
from sentence_transformers import SentenceTransformer

@lru_cache(maxsize=1)
def get_encoder() -> SentenceTransformer:
    """
    Load the BGE encoder once and cache it.
    """
    model_name = "BAAI/bge-base-en-v1.5"
    print(f"[encode] Loading encoder: {model_name}")
    return SentenceTransformer(model_name)

def encode_query(text: str) -> np.ndarray:
    """
    Encode a single query text into an embedding.
    Returns a numpy array of shape (embedding_dim,)
    """
    model = get_encoder()
    is_list = isinstance(text, list)
    embeddings = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embeddings = embeddings.astype(np.float32, copy=False)
    if not is_list:
        return embeddings
    else:
        return embeddings[0]
    
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="Query text to encode")
    args = ap.parse_args()

    vec = encode_query(args.query)
    print("Embedding shape:", vec.shape)
    print("First 5 dims:", vec[:5])
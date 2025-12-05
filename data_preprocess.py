#!/usr/bin/env python3
"""
data_preprocess.py
Reads MS MARCO docs, encodes with BGE, and outputs preprocessed_documents.json.

Usage:
  python data_preprocess.py \
    --docs documents.json \
    --out preprocessed_documents.json \
    --batch-size 64
"""

import argparse, json, sys, os
from typing import List, Dict, Any, Iterable
import numpy as np
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

def iter_docs(doc_path: str) -> Iterable[str]:
    """
    Robust loader:
    - If the file is a JSON array of strings: ["doc1", "doc2", ...]  -> yield each
    - If it's JSON lines (one JSON object or string per line) -> yield each
    - If it's raw text lines -> yield each nonempty line
    - If it's a JSON array of objects with a 'text' field -> use that
    """
    with open(doc_path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == "[":
            data = json.load(f)
            if isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, dict) and "text" in item:
                        yield str(item["text"])
                    else:
                        yield str(item)
            else:
                raise ValueError("Unexpected JSON structure in documents file.")
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "text" in obj:
                        yield str(obj["text"])
                    elif isinstance(obj, str):
                        yield obj
                    else:
                        yield str(obj)
                except json.JSONDecodeError:
                    yield line

def encode_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 64,
    normalize: bool = True
) -> np.ndarray:
    """
    Returns float32 np.ndarray of shape (N, 768)
    We normalize to unit length so L2 distance ~= cosine distance.
    """
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=True,
    )
    if emb.dtype != np.float32:
        emb = emb.astype(np.float32, copy=False)
    return emb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True, help="Path to documents.json downloaded")
    ap.add_argument("--out", default="preprocessed_documents.json",
                    help="Output file in the required format")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-normalize", action="store_true",
                    help="Disable L2-normalization of embeddings (default is normalize)")
    args = ap.parse_args()

    print("Loading documents...")
    docs = list(iter_docs(args.docs))
    if len(docs) == 0:
        print("No documents found. Check your input file.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(docs)} docs.")
    print("Loading encoder: BAAI/bge-base-en-v1.5 …")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    print("Encoding to 768-D embeddings…")
    embs = encode_texts(
        model,
        docs,
        batch_size=args.batch_size,
        normalize=(not args.no_normalize),
    )
    if embs.shape[1] != 768:
        raise RuntimeError(f"Expected 768 dims, got {embs.shape[1]}")

    print(f"Writing {args.out} …")
    out: List[Dict[str, Any]] = []
    for i, (text, vec) in enumerate(zip(docs, embs), start=0):
        out.append({
            "id": i,
            "text": text,
            "embedding": vec.tolist(),
        })

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    print(f"Done. Examples:\n  id=0 dim={len(out[0]['embedding'])}\n  total={len(out)}")

if __name__ == "__main__":
    main()
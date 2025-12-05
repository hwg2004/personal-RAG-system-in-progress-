#!/usr/bin/env python3
"""
vector_db.py
- Loads preprocessed_documents.json
- Builds a FAISS index (FlatL2 or IVFFlat) over embeddings
- Provides:
    VectorDB.search(query_emb, k)      -> (D, doc_ids)
    VectorDB.search_batch(query_embs)  -> (D, doc_ids) for a batch
    VectorDB.get_texts(doc_ids)        -> [text1, text2, ...]
- Still supports the Part 1 self-test.
"""

import argparse, json, sys
from typing import Tuple, List, Dict, Any
import numpy as np
import faiss


class VectorDB:
    def __init__(
        self,
        d: int = 768,
        index_type: str = "flat",
        nlist: int = 100,
        nprobe: int = 10,
    ):
        """
        index_type: "flat" (exact) or "ivfflat" (ANN)
        nlist     : number of IVF clusters (for ivfflat)
        nprobe    : number of clusters to probe at search time (for ivfflat)
        """
        if index_type not in {"flat", "ivfflat"}:
            raise ValueError("index_type must be 'flat' or 'ivfflat'.")

        self.d = d
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe

        # Index will be created after we load embeddings
        self.index: faiss.Index = None  # type: ignore

        self._id_by_row: np.ndarray | None = None
        self._embs: np.ndarray | None = None
        self._text_by_id: Dict[int, str] = {}

    # ---- internal helpers to build the FAISS index ----

    def _build_flat_index(self, embs: np.ndarray) -> None:
        index = faiss.IndexFlatL2(self.d)
        index.add(embs)
        self.index = index

    def _build_ivfflat_index(self, embs: np.ndarray) -> None:
        """
        Build an IVFFlat index using faiss.index_factory.
        """
        embs = np.ascontiguousarray(embs.astype("float32"))
        N = embs.shape[0]

        # keep nlist sane
        nlist = min(self.nlist, max(1, N // 100))
        print(f"[vector_db] Building IVFFlat with N={N}, nlist={nlist}, d={self.d}")

        # use index_factory instead of IndexIVFFlat directly
        index = faiss.index_factory(self.d, f"IVF{nlist},Flat", faiss.METRIC_L2)

        # train on all data (or subset if you want)
        index.train(embs)
        index.add(embs)

        # nprobe ≤ nlist
        nprobe = min(self.nprobe, nlist)
        index.nprobe = nprobe

        self.index = index
        # ---- public API ----

    def load_preprocessed(self, path: str) -> None:
        """
        Load preprocessed_documents.json and build:
        - self._embs       : (N, d) float32
        - self._id_by_row  : (N,) int64
        - self._text_by_id : dict[id] -> text
        - FAISS index      : FlatL2 or IVFFlat
        """
        with open(path, "r", encoding="utf-8") as f:
            data: List[Dict[str, Any]] = json.load(f)
        if not data:
            raise ValueError("Empty preprocessed file.")

        N = len(data)
        embs = np.zeros((N, self.d), dtype="float32")
        id_by_row = np.zeros((N,), dtype="int64")

        text_by_id: Dict[int, str] = {}

        for i, row in enumerate(data):
            emb = np.array(row["embedding"], dtype="float32")
            if emb.shape[0] != self.d:
                raise ValueError(f"Row {i} has dim {emb.shape[0]}, expected {self.d}")
            embs[i] = emb
            doc_id = int(row["id"])
            id_by_row[i] = doc_id
            text_by_id[doc_id] = row["text"]

        self._embs = embs
        self._id_by_row = id_by_row
        self._text_by_id = text_by_id

        # Build index according to index_type
        if self.index_type == "flat":
            self._build_flat_index(embs)
        else:
            self._build_ivfflat_index(embs)

    def search(self, query_emb: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        query_emb: shape (d,) or (1,d)
        Returns:
          D:   shape (1,k) L2 distances
          ids: shape (1,k) document IDs (NOT row indices)
        """
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1).astype("float32")
        else:
            query_emb = query_emb.astype("float32")
        assert query_emb.shape[1] == self.d

        D, I = self.index.search(query_emb, k)
        ids = self._id_by_row[I]
        return D, ids

    def search_batch(self, query_embs: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batched search using a simple Python loop over self.search().
        This avoids FAISS segfaults on multi-query search in this environment.
        query_embs: shape (B, d)
        Returns:
        D:   shape (B,k)
        ids: shape (B,k)
        """
        query_embs = np.asarray(query_embs, dtype="float32")
        assert query_embs.ndim == 2 and query_embs.shape[1] == self.d

        all_D = []
        all_ids = []
        for q in query_embs:
            D, ids = self.search(q, k=k)   # uses single-query FAISS search (which is stable)
            all_D.append(D[0])
            all_ids.append(ids[0])

        return np.stack(all_D, axis=0), np.stack(all_ids, axis=0)

    def get_texts(self, doc_ids: np.ndarray | List[int]) -> List[str]:
        """
        Given an array-like of doc_ids, return their texts in the same order.
        """
        if isinstance(doc_ids, np.ndarray):
            doc_ids = doc_ids.tolist()
        texts = []
        for did in doc_ids:
            did_int = int(did)
            txt = self._text_by_id.get(did_int, "")
            texts.append(txt)
        return texts

    def query_by_row(self, row: int, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convenience method: use one of the stored vectors as the query.
        """
        if self._embs is None:
            raise RuntimeError("Embeddings not loaded.")
        if row < 0 or row >= self._embs.shape[0]:
            raise IndexError("row out of range")
        q = self._embs[row].reshape(1, -1)
        return self.search(q, k=k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocessed", required=True,
                    help="Path to preprocessed_documents.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--self-test", action="store_true",
                    help="Run a quick self-similarity test (doc should retrieve itself first).")
    ap.add_argument("--query-doc", type=int,
                    help="Optional: use this row index as a query and print neighbors.")
    ap.add_argument("--index-type", choices=["flat", "ivfflat"], default="flat",
                    help="Type of FAISS index to build.")
    ap.add_argument("--nlist", type=int, default=100,
                    help="Number of IVF clusters (for ivfflat).")
    ap.add_argument("--nprobe", type=int, default=10,
                    help="Number of clusters to probe at search time (for ivfflat).")
    args = ap.parse_args()

    db = VectorDB(
        d=768,
        index_type=args.index_type,
        nlist=args.nlist,
        nprobe=args.nprobe,
    )
    print("Loading preprocessed embeddings …")
    db.load_preprocessed(args.preprocessed)
    print(f"FAISS Index size: {db.index.ntotal}")

    if args.self_test:
        D_row, I_row = db.index.search(db._embs[0:1], args.k)
        _, ids = db.query_by_row(0, k=args.k)
        print("Self-test using row=0")
        print("Distances:", D_row[0].tolist())
        print("FAISS rows:", I_row[0].tolist())
        print("Doc IDs:", ids[0].tolist())
        if I_row[0][0] == 0 and D_row[0][0] <= 1e-6:
            print("Passed: top-1 is itself with ~0 distance.")
        else:
            print("Failed: expected self at rank 1 with distance ~0.", file=sys.stderr)
            sys.exit(2)

    if args.query_doc is not None:
        row = args.query_doc
        D_row, I_row = db.index.search(db._embs[row:row+1], args.k)
        _, ids = db.query_by_row(row, k=args.k)
        print(f"\nQuery by stored row={row}")
        print("Distances:", D_row[0].tolist())
        print("FAISS rows:", I_row[0].tolist())
        print("Doc IDs:", ids[0].tolist())
        print("Texts:")
        for t in db.get_texts(ids[0]):
            print("----")
            print(t[:300].replace("\n", " "))
            if len(t) > 300:
                print("...")
#!/usr/bin/env python3
"""
main.py
Interactive RAG system + benchmarking:

- Loads preprocessed_documents.json into VectorDB
- Uses encode.encode_query to embed user questions
- Retrieves top-k documents
- Builds an augmented prompt
- Uses LLMGenerator to generate an answer

Benchmark mode additionally:
- Loads queries from queries.json
- Measures latency of:
    * document retrieval (encode + FAISS search)
    * question augmentation (build_augmented_prompt)
    * LLM generation
- Writes timings to a CSV file
"""

import argparse
import csv
import json
import time
from textwrap import shorten

from encode import encode_query
from vector_db import VectorDB
from llm_generation import LLMGenerator


def build_augmented_prompt(
    question: str,
    doc_texts: list[str],
) -> str:
    """
    Simple prompt format:
    Question + Top documents.
    This is our "question augmentation" step for timing purposes.
    """
    context_blocks = []
    for i, txt in enumerate(doc_texts, start=1):
        snippet = txt.strip()
        if len(snippet) > 1000:
            snippet = snippet[:1000] + " ..."
        context_blocks.append(f"Document {i}:\n{snippet}")

    context_str = "\n\n".join(context_blocks)

    prompt = f"""You are an assistant that answers questions using only the provided documents.

Question:
{question}

Relevant documents:
{context_str}

Using ONLY the information in the documents above, write a helpful answer to the question. 
If the documents are not sufficient, say you are not sure.
"""
    return prompt


def run_single_query(db: VectorDB, llm: LLMGenerator, question: str, k: int):
    """
    Run the RAG pipeline once, with timing for:
    - retrieval (encode + search)
    - question augmentation (prompt building)
    - generation (LLM answer)
    Returns (answer, timings_dict).
    """
    # -------- retrieval --------
    t0 = time.perf_counter()
    q_vec = encode_query(question)
    D, ids = db.search(q_vec, k=k)  # ids shape (1,k)
    top_ids = ids[0]
    top_texts = db.get_texts(top_ids)
    t1 = time.perf_counter()

    # -------- question augmentation (prompt building) --------
    prompt = build_augmented_prompt(question, top_texts)
    t2 = time.perf_counter()

    # -------- LLM generation --------
    answer = llm.generate(prompt, max_tokens=256)
    t3 = time.perf_counter()

    timings = {
        "T_retrieval": t1 - t0,
        "T_augmentation": t2 - t1,
        "T_generation": t3 - t2,
        "T_total": t3 - t0,
    }

    return answer, timings, (D, top_ids, top_texts)


def run_interactive(db: VectorDB, llm: LLMGenerator, k: int):
    print("\n[main] RAG system ready. Type a question, or 'quit' to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[main] Exiting.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("[main] Goodbye.")
            break

        print("[main] Running RAG pipeline…")
        answer, timings, (D, top_ids, _) = run_single_query(db, llm, question, k)

        print("[main] Retrieved doc IDs and distances:")
        for rank, (did, dist) in enumerate(zip(top_ids, D[0]), start=1):
            preview = shorten(
                db.get_texts([did])[0].replace("\n", " "), width=80
            )
            print(f"  {rank}. id={did}, dist={dist:.4f}, text≈\"{preview}\"")

        print(
            f"[timings] retrieval={timings['T_retrieval']:.4f}s, "
            f"augment={timings['T_augmentation']:.4f}s, "
            f"generation={timings['T_generation']:.4f}s, "
            f"total={timings['T_total']:.4f}s"
        )

        print("\nAssistant:", answer, "\n")


def run_benchmark(db: VectorDB, llm: LLMGenerator, args):
    """
    Runs the RAG pipeline over all queries in queries.json
    and writes timing breakdowns to a CSV.
    """
    print(f"[benchmark] Loading queries from {args.queries} …")
    with open(args.queries, "r") as f:
        raw = json.load(f)

    # handle either ["q1","q2",...] or [{"question": "..."}, ...]
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        queries = [item.get("question", "") for item in raw]
    else:
        queries = raw

    print(f"[benchmark] Loaded {len(queries)} queries.")
    print(f"[benchmark] Writing timings to {args.output_csv}")

    with open(args.output_csv, "w", newline="") as f:
        fieldnames = [
            "query",
            "k",
            "llm_model",
            "T_retrieval",
            "T_augmentation",
            "T_generation",
            "T_total",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for q in queries[:40]:
            answer, timings, _ = run_single_query(db, llm, q, args.k)
            writer.writerow({
                "query": q,
                "k": args.k,
                "llm_model": args.model,
                **timings,
            })

    print("[benchmark] Done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--preprocessed",
        default="preprocessed_documents.json",
        help="Path to preprocessed_documents.json",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=3,
        help="Top-k documents to retrieve",
    )
    ap.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="LLM model name passed to LLMGenerator",
    )
    ap.add_argument(
        "--mode",
        choices=["interactive", "benchmark"],
        default="interactive",
        help="Run interactive chat or benchmarking mode",
    )
    ap.add_argument(
        "--queries",
        default="queries.json",
        help="Path to queries.json (for benchmark mode)",
    )
    ap.add_argument(
        "--output_csv",
        default="timings_baseline.csv",
        help="Where to write timing results in benchmark mode",
    )
    ap.add_argument(
        "--index_type",
        choices=["flat", "ivfflat"],
        default="flat",
        help="FAISS index type to use",
    )
    ap.add_argument("--nlist", type=int, default=100)
    ap.add_argument("--nprobe", type=int, default=10)
    args = ap.parse_args()

    # 1. Load vector DB
    print("[main] Loading vector database…")
    db = VectorDB(
        d=768,
        index_type=args.index_type,
        nlist=args.nlist,
        nprobe=args.nprobe,
    )
    db.load_preprocessed(args.preprocessed)
    print(f"[main] Loaded {db.index.ntotal} documents into FAISS index.")

    # 2. Load LLM
    print("[main] Initializing LLM…")
    llm = LLMGenerator(model=args.model)

    if args.mode == "interactive":
        run_interactive(db, llm, k=args.k)
    else:
        run_benchmark(db, llm, args)


if __name__ == "__main__":
    main()
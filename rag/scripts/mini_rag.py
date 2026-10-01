"""A whole RAG pipeline in one file: chunk -> embed -> index -> hybrid search -> rerank -> prompt.

No framework, no vector database: the index is a normalised (N, d) tensor and the
search is one matmul. Companion to ../index.ipynb.

    python mini_rag.py --corpus ../..            --ask "how does layer-wise LR decay work?"
    python mini_rag.py --corpus /path/to/docs    --ask "..." --k 5 --no-rerank
    python mini_rag.py --corpus ../.. --store chroma --persist ./chroma_rag --ask "..."

Two index backends:
  numpy   brute-force matmul over an in-memory (N, d) matrix -- exact, rebuilt every run
  chroma  a persistent Chroma collection (HNSW, cosine) -- survives restarts, supports
          metadata filters and incremental upserts. `pip install chromadb`

Embedding backend (--embedder, default auto = first that works):
  1. ollama                 local server, e.g. `ollama pull qwen3-embedding:0.6b`
  2. sentence-transformers  (--st-model, default BAAI/bge-m3)
  3. a TF-IDF fallback (scikit-learn) so the script runs with no downloads at all.

Generation is optional: without --generate the assembled prompt is printed, with it the
prompt goes to a local Ollama model:

    python mini_rag.py --corpus ../.. --ask "..." --generate \
        --llm qwen3:30b-a3b-instruct-2507-q4_K_M

See ../../llama_cpp_gguf/index.ipynb for what these quantised local models are.
"""
import argparse
import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- chunking


def split_sections(text):
    """Markdown sections, ignoring '#' lines inside fenced code blocks."""
    sections, buf, fence = [], [], False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            fence = not fence
        if not fence and re.match(r"#{1,6} ", line) and buf:
            sections.append("\n".join(buf))
            buf = []
        buf.append(line)
    if buf:
        sections.append("\n".join(buf))
    return sections


def chunk_text(text, source, size=400, overlap=60, markdown=True):
    """Split on markdown headings first, then on a word budget, keeping a header trail."""
    out, trail = [], []
    sections = split_sections(text) if markdown else [text]
    for section in sections:
        m = re.match(r"(#{1,6}) (.+)", section) if markdown else None
        if m:
            level = len(m.group(1))
            trail = trail[: level - 1] + [m.group(2).strip()]
        header = " > ".join([source] + trail)
        words = section.split()
        step = max(size - overlap, 1)
        for i in range(0, max(len(words), 1), step):
            body = " ".join(words[i:i + size])
            if body.strip():
                out.append({"header": header, "text": f"{header}\n{body}", "source": source})
            if i + size >= len(words):
                break
    return out


def notebook_to_markdown(path):
    """Flatten a .ipynb into markdown: prose cells as-is, code cells fenced, outputs dropped."""
    nb = json.loads(path.read_text(errors="ignore"))
    parts = []
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", [])).strip()
        if not src:
            continue
        parts.append(src if cell.get("cell_type") == "markdown" else f"```python\n{src}\n```")
    return "\n\n".join(parts)


def load_corpus(root, patterns=("*.md", "*.py", "*.txt", "*.ipynb")):
    root = Path(root).resolve()
    chunks = []
    for pat in patterns:
        for path in sorted(root.rglob(pat)):
            if any(p in path.parts for p in (".git", "__pycache__", "node_modules")):
                continue
            try:
                text = (notebook_to_markdown(path) if path.suffix == ".ipynb"
                        else path.read_text(errors="ignore"))
            except (OSError, ValueError, KeyError):
                continue
            if text.strip():
                chunks += chunk_text(text, str(path.relative_to(root)),
                                     markdown=path.suffix in (".md", ".txt", ".ipynb"))
    return chunks


# --------------------------------------------------------------------------- embedding


def ollama_post(host, path, payload, timeout=600):
    req = urllib.request.Request(f"{host}{path}", method="POST",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class OllamaEmbedder:
    """Embeddings from a local Ollama server -- no Python model, no HF download."""

    # Qwen3-Embedding is trained with an instruction on the QUERY side only.
    QUERY_INSTRUCT = ("Instruct: Given a question, retrieve passages that answer it\n"
                      "Query: ")

    def __init__(self, model, host="http://localhost:11434", batch=32):
        self.model, self.host, self.batch = model, host, batch
        self.instruct = "qwen3-embedding" in model
        ollama_post(host, "/api/embed", {"model": model, "input": ["ping"]}, timeout=120)

    def __call__(self, texts, is_query=False):
        if is_query and self.instruct:
            texts = [self.QUERY_INSTRUCT + t for t in texts]
        out = []
        for i in range(0, len(texts), self.batch):
            r = ollama_post(self.host, "/api/embed",
                            {"model": self.model, "input": texts[i:i + self.batch]})
            out += r["embeddings"]
        v = np.asarray(out, dtype=np.float32)
        return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)


def ollama_generate(prompt, model, host="http://localhost:11434", num_ctx=8192, temperature=0.0):
    """One non-streaming chat turn. num_ctx must cover the whole retrieved context."""
    r = ollama_post(host, "/api/chat", {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_ctx": num_ctx, "temperature": temperature}})
    return r["message"]["content"], r.get("eval_count"), r.get("prompt_eval_count")


class STEmbedder:
    """sentence-transformers, with the query/passage asymmetry the model cards ask for."""

    def __init__(self, name):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name)
        self.e5 = "e5" in name.lower()

    def __call__(self, texts, is_query=False):
        if self.e5:                                   # E5-style models want explicit prefixes
            prefix = "query: " if is_query else "passage: "
            texts = [prefix + t for t in texts]
        v = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)


class TfidfEmbedder:
    """Zero-download fallback so the pipeline is runnable anywhere."""

    def __init__(self, corpus):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", max_features=50000)
        self.vec.fit(corpus)

    def __call__(self, texts, is_query=False):
        v = self.vec.transform(texts).toarray().astype(np.float32)
        return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)


# --------------------------------------------------------------------------- vector store


def chunk_id(c):
    """Content hash: re-indexing an unchanged file is a no-op, an edited one overwrites."""
    return hashlib.sha1(c["text"].encode()).hexdigest()[:16]


class ChromaStore:
    """A persistent Chroma collection, used exactly like the numpy matrix above."""

    def __init__(self, path, name="rag_corpus", space="cosine"):
        import chromadb
        self.client = chromadb.PersistentClient(path=path)
        self.col = self.client.get_or_create_collection(
            name, metadata={"hnsw:space": space})      # cosine, not the default L2

    def index(self, chunks, embed, batch=256):
        """Upsert only the chunks whose content hash is not in the collection yet."""
        ids = [chunk_id(c) for c in chunks]
        known = set(self.col.get(ids=ids, include=[])["ids"])
        todo = [(i, c) for i, c in zip(ids, chunks) if i not in known]
        for s in range(0, len(todo), batch):
            part = todo[s:s + batch]
            self.col.upsert(
                ids=[i for i, _ in part],
                embeddings=embed([c["text"] for _, c in part]).tolist(),
                documents=[c["text"] for _, c in part],
                metadatas=[{"source": c["source"], "header": c["header"]} for _, c in part])
        return len(todo), self.col.count()

    def search(self, qvec, k, where=None):
        r = self.col.query(query_embeddings=qvec.tolist(), n_results=k,
                           where=where, include=["documents", "metadatas", "distances"])
        hits = []
        for cid, doc, meta, dist in zip(r["ids"][0], r["documents"][0],
                                        r["metadatas"][0], r["distances"][0]):
            hits.append({"id": cid, "text": doc, "header": meta["header"],
                         "source": meta["source"], "score": 1.0 - dist})   # cosine distance
        return hits


# --------------------------------------------------------------------------- retrieval


def dense_search(index, qvec, k):
    scores = (qvec @ index.T)[0]                      # unit vectors -> cosine == dot
    top = np.argsort(-scores)[:k]
    return list(top), scores


def bm25_search(chunks, query, k, k1=1.5, b=0.75):
    """Plain BM25 over whitespace tokens: catches exact names, flags, error codes."""
    tok = lambda s: re.findall(r"[a-z0-9_]+", s.lower())
    docs = [tok(c["text"]) for c in chunks]
    avgdl = sum(len(d) for d in docs) / max(len(docs), 1)
    df = Counter(t for d in docs for t in set(d))
    n = len(docs)
    scores = np.zeros(n, dtype=np.float32)
    for term in set(tok(query)):
        if term not in df:
            continue
        idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
        for i, d in enumerate(docs):
            f = d.count(term)
            if f:
                scores[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(d) / avgdl))
    return list(np.argsort(-scores)[:k]), scores


def rrf(ranked_lists, k=60):
    """Reciprocal rank fusion: no score calibration needed between dense and BM25."""
    fused = {}
    for lst in ranked_lists:
        for rank, doc in enumerate(lst):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused, key=fused.get, reverse=True)


def rerank(query, chunks, ids, model_name, top):
    """Cross-encoder over the fused shortlist only -- accurate, and far too slow for the index."""
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(model_name)
    pairs = [[query, chunks[i]["text"]] for i in ids]
    scores = ce.predict(pairs)
    order = np.argsort(-np.asarray(scores))[:top]
    return [ids[j] for j in order]


# --------------------------------------------------------------------------- prompt


PROMPT = """Answer the question using ONLY the context below.
If the context does not contain the answer, say you don't know.
Cite the [n] of every chunk you use.

{context}

Question: {question}
"""


def build_prompt(question, chunks, ids):
    blocks = [f"[{n}] ({chunks[i]['header']})\n{chunks[i]['text']}"
              for n, i in enumerate(ids, 1)]
    return PROMPT.format(context="\n\n".join(blocks), question=question)


def pick_embedder(args, texts):
    """Ollama -> sentence-transformers -> TF-IDF, unless --embedder forces one."""
    want = args.embedder
    if want in ("auto", "ollama"):
        try:
            e = OllamaEmbedder(args.ollama_embed, args.host)
            print(f"embedder: ollama {args.ollama_embed}")
            return e
        except Exception as exc:
            if want == "ollama":
                raise
            print(f"ollama embedder unavailable ({type(exc).__name__})")
    if want in ("auto", "st"):
        try:
            e = STEmbedder(args.st_model)
            print(f"embedder: {args.st_model}")
            return e
        except Exception as exc:
            if want == "st":
                raise
            print(f"sentence-transformers unavailable ({type(exc).__name__})")
    print("embedder: TF-IDF fallback")
    return TfidfEmbedder(texts)


# --------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--ask", required=True)
    ap.add_argument("--k", type=int, default=5, help="chunks in the final prompt")
    ap.add_argument("--shortlist", type=int, default=50, help="candidates before reranking")
    ap.add_argument("--embedder", choices=["auto", "ollama", "st", "tfidf"], default="auto")
    ap.add_argument("--ollama-embed", default="qwen3-embedding:0.6b")
    ap.add_argument("--st-model", default="BAAI/bge-m3", help="sentence-transformers model")
    ap.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--generate", action="store_true", help="answer with a local Ollama model")
    ap.add_argument("--llm", default="qwen3:30b-a3b-instruct-2507-q4_K_M")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--store", choices=["numpy", "chroma"], default="numpy")
    ap.add_argument("--persist", default="./chroma_rag", help="Chroma directory (--store chroma)")
    ap.add_argument("--where-source", help="restrict to sources containing this string "
                                           "(Chroma metadata filter)")
    args = ap.parse_args()

    chunks = load_corpus(args.corpus)
    print(f"{len(chunks)} chunks from {args.corpus}")
    if not chunks:
        return

    texts = [c["text"] for c in chunks]
    embed = pick_embedder(args, texts)

    qvec = embed([args.ask], is_query=True)

    if args.store == "chroma":
        if isinstance(embed, TfidfEmbedder):
            print("warning: TF-IDF vectors are refitted per corpus, so a persisted Chroma "
                  "collection is only comparable within one run -- use a real embedding model")
        store = ChromaStore(args.persist)
        added, total = store.index(chunks, embed)
        print(f"chroma: +{added} new chunks, {total} in collection at {args.persist}")
        by_id = {chunk_id(c): i for i, c in enumerate(chunks)}
        where = {"source": {"$contains": args.where_source}} if args.where_source else None
        hits = store.search(qvec, args.shortlist, where=where)
        dense_ids = [by_id[h["id"]] for h in hits if h["id"] in by_id]
    else:
        index = embed(texts)                              # (N, d), L2-normalised
        dense_ids, _ = dense_search(index, qvec, args.shortlist)

    sparse_ids, _ = bm25_search(chunks, args.ask, args.shortlist)
    ids = rrf([dense_ids, sparse_ids])[: args.shortlist]

    if not args.no_rerank:
        try:
            ids = rerank(args.ask, chunks, ids, args.reranker, args.k)
        except Exception as exc:
            print(f"reranker unavailable ({type(exc).__name__}), using fused order")
            ids = ids[: args.k]
    else:
        ids = ids[: args.k]

    print("\nretrieved:")
    for n, i in enumerate(ids, 1):
        print(f"  [{n}] {chunks[i]['header']}")

    prompt = build_prompt(args.ask, chunks, ids)
    if not args.generate:
        print("\n" + "=" * 70 + "\n" + prompt)
        return

    print(f"\ngenerating with {args.llm} (num_ctx {args.num_ctx}) ...")
    answer, out_tok, in_tok = ollama_generate(prompt, args.llm, args.host, args.num_ctx)
    print(f"prompt {in_tok} tok -> {out_tok} tok\n" + "=" * 70 + "\n" + answer)


if __name__ == "__main__":
    main()

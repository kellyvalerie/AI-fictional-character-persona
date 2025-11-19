import os
import json
import logging
from difflib import SequenceMatcher
import pickle
import requests
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


WORKING_DIR = os.path.join(os.path.dirname(__file__), "source_data")
WORKING_DIR = os.path.normpath(WORKING_DIR) + os.sep
entity_graph = json.load(open(os.path.join(WORKING_DIR, "entity_graph.json")))

EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")  # 5 GB RAM max, CPU-fast
INDEX_FILE = os.path.join(WORKING_DIR, "faiss_index.bin")
METADATA_FILE = os.path.join(WORKING_DIR, "chunks_metadata.pkl")

def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def best_match(name, candidates, threshold=0.6):
    best = None
    best_score = 0.0
    for c in candidates:
        s = similar(name.lower(), c.lower())
        if s > best_score:
            best_score = s
            best = c
    if best_score >= threshold:
        return best, best_score
    return None, best_score


def load_data():
    rel_path = os.path.join(WORKING_DIR, "entities_relationships.json")
    dlg_path = os.path.join(WORKING_DIR, "entity_dialogues.json")
    graph_path = os.path.join(WORKING_DIR, "entity_graph.json")

    rels = []
    dlg = []
    nodes = []
    if os.path.exists(rel_path):
        with open(rel_path, "r", encoding="utf-8") as f:
            rels = json.load(f)
    if os.path.exists(dlg_path):
        with open(dlg_path, "r", encoding="utf-8") as f:
            dlg = json.load(f)
    if os.path.exists(graph_path):
        with open(graph_path, "r", encoding="utf-8") as f:
            g = json.load(f)
            nodes = [n["id"] for n in g.get("nodes", [])]

    return rels, dlg, nodes

def build_hybrid_index():
        if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
            logging.info("Loading pre-built hybrid index...")
            index = faiss.read_index(INDEX_FILE)
            with open(METADATA_FILE, "rb") as f:
                chunks = pickle.load(f)
            return index, chunks

        logging.info("Building new hybrid index...")
        rels, dlg, nodes = load_data()
        graph = json.load(open(os.path.join(WORKING_DIR, "entity_graph.json")))

        # Create rich chunks
        chunks = []
        # 1. Dialogue chunks (best for voice)
        for d in dlg:
            text = d.get("dialogue") or d.get("context") or ""
            if not text.strip():
                continue
            speaker = d.get("speaker", "Unknown")
            # Resolve canonical name
            canon = next((n["id"] for n in graph["nodes"] if speaker in n["members"]), speaker)
            chunks.append({
                "text": text.strip(),
                "type": "dialogue",
                "speaker": canon,
                "original_speaker": speaker,
                "entities": [canon]
            })

        # 2. Relationship chunks (best for facts)
        for r in rels:
            e1 = r.get("entity1")
            e2 = r.get("entity2")
            if not e1 or not e2:
                continue
            # Resolve canonical
            c1 = next((n["id"] for n in graph["nodes"] if e1 in n["members"]), e1)
            c2 = next((n["id"] for n in graph["nodes"] if e2 in n["members"]), e2)
            text = f"{c1} {r.get('relationship', 'interacts with')} {c2}. {r.get('context', '')}".strip()
            chunks.append({
                "text": text,
                "type": "relationship",
                "entities": [c1, c2],
                "relationship": r.get("relationship")
            })

        texts = [c["text"] for c in chunks]
        embeddings = EMBEDDER.encode(texts, batch_size=32, show_progress_bar=True)
        
        # Build FAISS index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner product = cosine after normalization
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        index.add(embeddings.astype('float32'))

        # Save
        faiss.write_index(index, INDEX_FILE)
        with open(METADATA_FILE, "wb") as f:
            pickle.dump(chunks, f)
        
        logging.info(f"Built index with {len(chunks)} chunks")
        return index, chunks


# def build_vectorizers(rels, dlg):
#     # Build text corpora
#     rel_texts = []
#     rel_meta = []
#     for r in rels:
#         text = " ".join([str(r.get("entity1", "")), str(r.get("entity2", "")), str(r.get("relationship", "")), str(r.get("context", ""))])
#         rel_texts.append(text)
#         rel_meta.append(r)

#     dlg_texts = []
#     dlg_meta = []
#     for d in dlg:
#         txt = d.get("dialogue") or d.get("context") or ""
#         speaker = d.get("speaker")
#         dlg_texts.append(txt)
#         dlg_meta.append(d)

#     all_texts = rel_texts + dlg_texts
#     vectorizer = TfidfVectorizer()
#     if not all_texts:
#         # create empty matrices for consistent return shape
#         X = vectorizer.fit_transform([""])
#         rel_X = X[:0]
#         dlg_X = X[0:]
#         return vectorizer, (rel_X, dlg_X), rel_texts, dlg_texts, rel_meta, dlg_meta

#     vectorizer = TfidfVectorizer().fit(all_texts)
#     X = vectorizer.transform(all_texts)

#     rel_X = X[: len(rel_texts)] if rel_texts else X[:0]
#     dlg_X = X[len(rel_texts) :] if dlg_texts else X[0:]

#     return vectorizer, (rel_X, dlg_X), rel_texts, dlg_texts, rel_meta, dlg_meta

# # set k = 5
# def topk_sim(query, vectorizer, X, k=5):
#     if X is None or X.shape[0] == 0:
#         return []
#     qv = vectorizer.transform([query])
#     sims = (X @ qv.T).toarray().ravel()
#     idx = np.argsort(-sims)[:k]
#     return list(zip(idx.tolist(), sims[idx].tolist()))
def hybrid_retrieve(query, persona, index, chunks, k=8, graph=None):
    if graph is None:
        graph = json.load(open(os.path.join(WORKING_DIR, "entity_graph.json")))

    # 1. Vector search
    q_emb = EMBEDDER.encode([query])
    q_emb = q_emb / np.linalg.norm(q_emb)
    scores, indices = index.search(q_emb.astype('float32'), k*3)

    # 2. Graph boosting: prioritize chunks involving persona or close allies
    boosted = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        boost = 1.0
        if persona in chunk["entities"]:
            boost += 0.4
        # Boost if involves close relationship
        for e in graph["edges"]:
            if persona in (e["source"], e["target"]) and any(x in chunk["entities"] for x in (e["source"], e["target"])):
                if e["count"] > 3:
                    boost += 0.3
        boosted.append((score * boost, idx, chunk))

    boosted.sort(key=lambda x: -x[0])
    return [item[2] for item in boosted[:k]]

def interactive(k=5, inference_mode=False):
    # At start of interactive()
    index, chunks = build_hybrid_index()
    graph = json.load(open(os.path.join(WORKING_DIR, "entity_graph.json")))
    rels, dlg, nodes = load_data()
    if not rels and not dlg:
        print("No relationship or dialogue data found in source_data/. Run preprocessing first.")
        return

    if TfidfVectorizer is None or np is None:
        print("Missing dependencies: scikit-learn and/or numpy not installed.\nInstall with: pip install -r requirements.txt")
        return

    vectorizer, (rel_X, dlg_X), rel_texts, dlg_texts, rel_meta, dlg_meta = build_vectorizers(rels, dlg)


    chosen = input("Which character do you want? ").strip()
    match, score = best_match(chosen, nodes)
    
    if match is None:
        print(f"No close match found for '{chosen}'. Proceeding with exact string as persona.")
        persona = chosen
    else:
        persona = match

    print("Start asking questions (type 'quit' or 'exit' to stop).")
    while True:
        q = input("Q: ").strip()
        if q.lower() in ("quit", "exit"):
            print("Goodbye")
            break

        # # Identify entity mentions in question: try to match nodes
        # mentioned = []
        # for n in nodes:
        #     if n.lower() in q.lower() or similar(n, q) > 0.8:
        #         mentioned.append(n)

        # if not mentioned:
        #     # default to chosen persona
        #     mentioned = [persona]

        # print(f"Identified entities (for query): {mentioned}")

        # retrieve top K from relationships and dialogues
        # rel_results = []
        # dlg_results = []
        # if rel_X is not None:
        #     rel_results = topk_sim(q, vectorizer, rel_X, k=k)
        # if dlg_X is not None:
        #     dlg_results = topk_sim(q, vectorizer, dlg_X, k=k)

        # print(f"Top {k} related relationship contexts:")
        # for idx, score in rel_results:
        #     meta = rel_meta[idx]
        #     print(f"- [{score:.3f}] {meta.get('entity1')} - {meta.get('relationship')} - {meta.get('entity2')}: {meta.get('context')}")

        # print(f"\nTop {k} related dialogue snippets:")
        # for idx, score in dlg_results:
        #     meta = dlg_meta[idx]
        #     print(f"- [{score:.3f}] {meta.get('speaker')}: {meta.get('dialogue') or meta.get('context')}")

        # Inside loop, replace everything after input q:
        results = hybrid_retrieve(q, persona, index, chunks, k=6, graph=graph)

        support_texts = []
        for r in results:
            prefix = f"[{r['type'].upper()}]"
            if r['type'] == 'dialogue':
                prefix += f" {r['original_speaker'] or r['speaker']}:"
            support_texts.append(f"{prefix} {r['text']}")

        support_block = "\n\n".join(support_texts)

        # Combine contexts for LLM (just print combined context for now)
        combined = []
        for idx, _ in rel_results:
            combined.append(rel_texts[idx])
        for idx, _ in dlg_results:
            combined.append(dlg_texts[idx])

        print("\n=== Combined context (for LLM) ===")
        for i, c in enumerate(combined[: k * 2]):
            print(f"[{i+1}] {c}\n")
        print("=== End context ===\n")

        # Attempt to generate an answer from an LLM (Ollama) using only the combined context
        try:
            model = os.environ.get("OLLAMA_MODEL", "gemma2:2b")
            host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            # Build support list from top results
            sources = []
            for idx, score in rel_results:
                sources.append(("relation", idx, score, rel_texts[idx]))
            for idx, score in dlg_results:
                sources.append(("dialogue", idx, score, dlg_texts[idx]))
            sources.sort(key=lambda x: -x[2])

            if not sources:
                print("No supporting context found  skipping LLM and replying 'I don't know'.")
                print(f"\n--- {persona} (LLM answer) ---")
                print("I don't know")
                print("--- end LLM answer ---\n")
                continue

            # Keyword boosting: prefer supports that contain tokens from the question
            q_tokens = {w.strip('.,?').lower() for w in q.split()}
            preferred = None
            for s in sources:
                text = (s[3] or "").lower()
                if any(tok in text for tok in q_tokens):
                    preferred = s
                    break
            if preferred:
                sources = [preferred] + [s for s in sources if s is not preferred]

            # take top N supports
            top_n = max(3, min(5, len(sources)))
            support_texts = [s[3] for s in sources[:top_n]]
            support_block = "\n\n---\n\n".join(support_texts)

            # Build prompt depending on inference mode
            if inference_mode:
                prompt = (
                    f"You are {persona}.\n"
                    "Use the SUPPORT snippets below. If the exact answer is present in the SUPPORT, answer directly from it.\n"
                    "If the exact answer is NOT present in the SUPPORT, reply with a single line that starts with:\n"
                    "INFERENCE: followed by your best inferred answer based ONLY on the SUPPORT snippets. Do not invent unrelated facts.\n\n"
                    "SUPPORT SNIPPETS (top {}):\n".format(len(support_texts))
                    + support_block
                    + "\n\nQUESTION:\n"
                    + q
                    + f"\n\nAnswer as {persona}:"
                )
            else:
                prompt = (
                    f"You are {persona}.\n"
                    "Use ONLY the SUPPORT snippets below to answer the question. Do not add facts that are not present in the SUPPORT. "
                    "If the exact answer cannot be produced from the SUPPORT, reply EXACTLY: I don't know\n\n"
                    "SUPPORT SNIPPETS (top {}):\n".format(len(support_texts))
                    + support_block
                    + "\n\nQUESTION:\n"
                    + q
                    + f"\n\nAnswer as {persona}:"
                )

            # call ollama (streaming preferred)
            try:
                resp = None
                payload = {"model": model, "prompt": prompt, "temperature": 0, "max_tokens": 256}
                timeout = int(os.environ.get("OLLAMA_TIMEOUT", "120"))
                streamed = False
                try:
                    with requests.post(f"{host}/api/generate", json=payload, timeout=timeout, stream=True) as r:
                        r.raise_for_status()
                        collected = []
                        printed_fragments = False
                        for ln in r.iter_lines(decode_unicode=True):
                            if not ln:
                                continue
                            try:
                                obj = json.loads(ln)
                                if isinstance(obj, dict) and obj.get("response") is not None:
                                    frag = obj.get("response") or ""
                                    print(frag, end="", flush=True)
                                    collected.append(str(frag))
                                    printed_fragments = True
                                if isinstance(obj, dict) and obj.get("done"):
                                    break
                            except Exception:
                                # non-JSON line: print and collect
                                print(ln, end="", flush=True)
                                collected.append(str(ln))
                                printed_fragments = True
                        if printed_fragments:
                            print("\n--- end LLM answer ---\n")
                            streamed = True
                        resp = "".join(collected) if collected else None
                except Exception:
                    # Fallback: non-streaming
                    r = requests.post(f"{host}/api/generate", json=payload, timeout=timeout)
                    r.raise_for_status()
                    text = r.text
                    try:
                        resp = json.loads(text)
                    except Exception:
                        resp = text.strip()

                # If strict mode produced 'I don't know' and inference_mode is enabled, do a second inference call
                resp_text = ""
                if isinstance(resp, (list, dict)):
                    resp_text = json.dumps(resp, ensure_ascii=False)
                elif isinstance(resp, str):
                    resp_text = resp

                simple_unk = resp_text.strip().lower()
                if (not inference_mode) and streamed:
                    # already printed streaming output in strict mode; nothing more to do
                    continue

                if inference_mode and (simple_unk in {"i don't know", "i dont know", "i don't know."}):
                    # build an inference prompt that asks for INFERENCE:
                    infer_prompt = (
                        f"You are {persona}.\n"
                        "Using ONLY the SUPPORT SNIPPETS below, if the exact answer is present, answer from it. "
                        "If not present, reply with a single line starting with 'INFERENCE:' followed by your best inference based ONLY on the SUPPORT snippets. Do not invent unrelated facts.\n\n"
                        "SUPPORT SNIPPETS (top {}):\n".format(len(support_texts))
                        + support_block
                        + "\n\nQUESTION:\n"
                        + q
                        + f"\n\nAnswer as {persona}:"
                    )
                    try:
                        r2 = requests.post(f"{host}/api/generate", json={"model": model, "prompt": infer_prompt, "temperature": 0, "max_tokens": 256}, timeout=timeout)
                        r2.raise_for_status()
                        text2 = r2.text
                        try:
                            infer_resp = json.loads(text2)
                            infer_text = infer_resp if isinstance(infer_resp, str) else json.dumps(infer_resp, ensure_ascii=False)
                        except Exception:
                            infer_text = text2.strip()
                        print("\n--- INFERENCE (fallback) ---")
                        print(infer_text)
                        print("--- end INFERENCE ---\n")
                    except Exception as e:
                        logging.warning("Inference call failed: %s", e)

                # Only print assembled response when it wasn't already streamed
                if not streamed:
                    print(f"\n--- {persona} (LLM answer) ---")
                    if isinstance(resp, (list, dict)):
                        print(json.dumps(resp, ensure_ascii=False, indent=2))
                    else:
                        print(resp)
                    print("--- end LLM answer ---\n")
            except Exception as e:
                logging.warning("LLM call failed: %s", e)
                print("LLM unavailable or failed to generate an answer. (Install/run Ollama and set OLLAMA_HOST/OLLAMA_MODEL if desired)")
        except Exception as e:
            logging.exception("Error while attempting LLM answer: %s", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive persona query tool")
    parser.add_argument("--k", type=int, default=5, help="number of top results to retrieve")
    parser.add_argument("--infer", action="store_true", help="allow the model to infer when evidence is incomplete (prefix inferences with 'INFERENCE:' )")
    args = parser.parse_args()

    try:
        interactive(k=args.k, inference_mode=args.infer)
    except Exception as e:
        logging.exception("Query tool failed: %s", e)

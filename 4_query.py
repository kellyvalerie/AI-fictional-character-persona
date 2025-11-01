import os
import json
import logging
from difflib import SequenceMatcher
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


WORKING_DIR = os.path.join(os.path.dirname(__file__), "source_data")
WORKING_DIR = os.path.normpath(WORKING_DIR) + os.sep


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


def build_vectorizers(rels, dlg):
    # Build text corpora
    rel_texts = []
    rel_meta = []
    for r in rels:
        text = " ".join([str(r.get("entity1", "")), str(r.get("entity2", "")), str(r.get("relationship", "")), str(r.get("context", ""))])
        rel_texts.append(text)
        rel_meta.append(r)

    dlg_texts = []
    dlg_meta = []
    for d in dlg:
        txt = d.get("dialogue") or d.get("context") or ""
        speaker = d.get("speaker")
        dlg_texts.append(txt)
        dlg_meta.append(d)

    all_texts = rel_texts + dlg_texts
    if not all_texts:
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform([""])
        return vectorizer, X, rel_texts, dlg_texts, rel_meta, dlg_meta

    vectorizer = TfidfVectorizer().fit(all_texts)
    X = vectorizer.transform(all_texts)

    rel_X = X[: len(rel_texts)] if rel_texts else None
    dlg_X = X[len(rel_texts) :] if dlg_texts else None

    return vectorizer, (rel_X, dlg_X), rel_texts, dlg_texts, rel_meta, dlg_meta

# set k = 5
def topk_sim(query, vectorizer, X, k=5):
    if X is None or X.shape[0] == 0:
        return []
    qv = vectorizer.transform([query])
    sims = (X @ qv.T).toarray().ravel()
    idx = np.argsort(-sims)[:k]
    return list(zip(idx.tolist(), sims[idx].tolist()))


def interactive(k=5):
    rels, dlg, nodes = load_data()
    if not rels and not dlg:
        print("No relationship or dialogue data found in source_data/. Run preprocessing first.")
        return

    if TfidfVectorizer is None or np is None:
        print("Missing dependencies: scikit-learn and/or numpy not installed.\nInstall with: pip install -r requirements.txt")
        return

    vectorizer, (rel_X, dlg_X), rel_texts, dlg_texts, rel_meta, dlg_meta = build_vectorizers(rels, dlg)

    # print("Available characters (from entity_graph):")
    # for n in nodes:
    #     print(" - ", n)

    chosen = input("Which character do you want? ").strip()
    match, score = best_match(chosen, nodes)
    
    if match is None:
        print(f"No close match found for '{chosen}'. Proceeding with exact string as persona.")
        persona = chosen
        interactive(k=5)
    else:
        persona = match
        print(f"Using persona: {persona} (match score {score:.2f})")

    print("Start asking questions (type 'quit' or 'exit' to stop).")
    while True:
        q = input("Q: ").strip()
        if q.lower() in ("quit", "exit"):
            print("Goodbye")
            break

        # Identify entity mentions in question: try to match nodes
        mentioned = []
        for n in nodes:
            if n.lower() in q.lower() or similar(n, q) > 0.8:
                mentioned.append(n)

        if not mentioned:
            # default to chosen persona
            mentioned = [persona]

        print(f"Identified entities (for query): {mentioned}")

        # retrieve top K from relationships and dialogues
        rel_results = []
        dlg_results = []
        if rel_X is not None:
            rel_results = topk_sim(q, vectorizer, rel_X, k=k)
        if dlg_X is not None:
            dlg_results = topk_sim(q, vectorizer, dlg_X, k=k)

        print(f"Top {k} related relationship contexts:")
        for idx, score in rel_results:
            meta = rel_meta[idx]
            print(f"- [{score:.3f}] {meta.get('entity1')} - {meta.get('relationship')} - {meta.get('entity2')}: {meta.get('context')}")

        print(f"\nTop {k} related dialogue snippets:")
        for idx, score in dlg_results:
            meta = dlg_meta[idx]
            print(f"- [{score:.3f}] {meta.get('speaker')}: {meta.get('dialogue') or meta.get('context')}")

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
            model = os.environ.get("OLLAMA_MODEL", "phi3:mini")
            host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            # Build a strict prompt that instructs the model to answer only from context
            # Choose the strongest supporting snippet and build a strict prompt
            sources = []
            for idx, score in rel_results:
                sources.append(("relation", idx, score, rel_texts[idx]))
            for idx, score in dlg_results:
                sources.append(("dialogue", idx, score, dlg_texts[idx]))
            sources.sort(key=lambda x: -x[2])

            if not sources:
                print("No supporting context found — skipping LLM and replying 'I don't know'.")
                print(f"\n--- {persona} (LLM answer) ---")
                print("I don't know")
                print("--- end LLM answer ---\n")
                continue

            # pick top support for strict, evidence-based answering
            top_kind, top_idx, top_score, top_text = sources[0]
            # debug: show which snippet we're sending
            logging.info("Top support: kind=%s idx=%s score=%.4f", top_kind, top_idx, top_score)
            print(f"\nUsing top support (score={top_score:.3f}, kind={top_kind}, idx={top_idx}) for the LLM call.\n")

            prompt = (
                f"You are {persona}.\n"
                "Use ONLY the SUPPORT text below to answer the question. Do not add any information that is not present in the SUPPORT.\n"
                "If the answer cannot be produced exactly from the SUPPORT, reply EXACTLY: I don't know\n\n"
                "SUPPORT:\n"
                + top_text
                + "\n\nQUESTION:\n"
                + q
                + f"\n\nAnswer as {persona}:"
            )
            # call ollama
            try:
                resp = None
                payload = {"model": model, "prompt": prompt, "temperature": 0, "max_tokens": 256}
                timeout = int(os.environ.get("OLLAMA_TIMEOUT", "120"))
                streamed = False
                # Stream the response so we can display incremental fragments as they arrive
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
                    # Fallback: non-streaming request (older behavior)
                    r = requests.post(f"{host}/api/generate", json=payload, timeout=timeout)
                    r.raise_for_status()
                    text = r.text
                    try:
                        resp = json.loads(text)
                    except Exception:
                        resp = text.strip()
                # Only print assembled response when it wasn't already streamed
                if not streamed:
                    print(f"\n--- {persona} (LLM answer) ---")
                    if isinstance(resp, (list, dict)):
                        print(json.dumps(resp, ensure_ascii=False, indent=2))
                    else:
                        print(resp)
                    # print("--- end LLM answer ---\n")
            except Exception as e:
                logging.warning("LLM call failed: %s", e)
                print("LLM unavailable or failed to generate an answer. (Install/run Ollama and set OLLAMA_HOST/OLLAMA_MODEL if desired)")
        except Exception as e:
            logging.exception("Error while attempting LLM answer: %s", e)


if __name__ == "__main__":
    try:
        interactive(k=5)
    except Exception as e:
        logging.exception("Query tool failed: %s", e)

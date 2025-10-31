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

    print("Available characters (from entity_graph):")
    for n in nodes:
        print(" - ", n)

    chosen = input("Which character do you want? ").strip()
    match, score = best_match(chosen, nodes)
    if match is None:
        print(f"No close match found for '{chosen}'. Proceeding with exact string as persona.")
        persona = chosen
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
            prompt = (
                f"You are {persona}.\n"
                "Answer the user question using ONLY the CONTEXT below. Do not add any information that is not present in the context. If the answer is not contained in the context, reply 'I don't know'. Keep the answer in-character for the persona.\n\n"
                "CONTEXT:\n"
                + "\n\n".join(combined)
                + "\n\nQUESTION:\n"
                + q
                + f"\n\nAnswer as {persona}:"
            )
            # call ollama
            try:
                resp = None
                payload = {"model": model, "prompt": prompt}
                timeout = int(os.environ.get("OLLAMA_TIMEOUT", "120"))
                r = requests.post(f"{host}/api/generate", json=payload, timeout=timeout)
                # r = requests.post(f"{host}/api/generate", json=payload, timeout=30)
                r.raise_for_status()
                text = r.text.strip()
                try:
                    resp = json.loads(text)
                except Exception:
                    resp = text

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
    try:
        interactive(k=5)
    except Exception as e:
        logging.exception("Query tool failed: %s", e)

import os
import json
import re
import logging
from collections import defaultdict, Counter
from difflib import SequenceMatcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

WORKING_DIR = os.path.join(os.path.dirname(__file__), "source_data")
WORKING_DIR = os.path.normpath(WORKING_DIR) + os.sep


def normalize_name(name: str) -> str:
    if not name:
        return ""
    s = name.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


class DSU:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, a):
        p = self.parent
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        self.parent[rb] = ra


def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def are_synonyms(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    if a in b or b in a:
        return True
    a_tokens = a.split()
    b_tokens = b.split()
    if len(a_tokens) >= 2 and len(b_tokens) >= 1 and a_tokens[-1] == b_tokens[-1]:
        return True
    if len(b_tokens) >= 2 and len(a_tokens) >= 1 and b_tokens[-1] == a_tokens[-1]:
        return True
    if similar(a, b) >= 0.86:
        return True
    return False


def build_dialogue_graph(input_path=None, output_path=None, verbose=True):
    if input_path is None:
        input_path = os.path.join(WORKING_DIR, "entity_dialogues.json")
    if output_path is None:
        output_path = os.path.join(WORKING_DIR, "entity_dialogue_graph.json")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # remove existing output if present
    try:
        if os.path.exists(output_path):
            os.remove(output_path)
            if verbose:
                logging.info(f"Removed existing output file: {output_path}")
    except Exception:
        logging.debug("Could not remove existing output file (continuing)")

    with open(input_path, "r", encoding="utf-8") as f:
        dialogues = json.load(f)

    # collect speakers and counts (preserve original speaker strings to keep conversational character)
    speakers = []
    freq = Counter()
    for d in dialogues:
        s = d.get("speaker")
        if s:
            speakers.append(s)
            freq[s] += 1

    uniques = list(dict.fromkeys(speakers))
    if verbose:
        logging.info(f"Found {len(speakers)} dialogue mentions, {len(uniques)} unique speakers (preserved)")

    # Preserve speaker surface forms exactly (no clustering) to maintain dialogue character.
    canonical = {u: u for u in uniques}
    clusters = {u: [u] for u in uniques}

    # Build interactions: group dialogues by 'context' (if context string exists and is shared)
    # For each context, link all speakers appearing in that context (all pairs). If context is empty
    # or too generic, fall back to adjacency linking in the dialogues sequence.
    context_map = defaultdict(list)
    for d in dialogues:
        ctx = d.get("context") or ""
        speaker = d.get("speaker")
        
        # prefer explicit 'dialogue' field for the quoted text, otherwise use context
        dialogue_text = d.get("dialogue") if d.get("dialogue") is not None else d.get("context")
        if not speaker:
            continue
        context_map[ctx].append((speaker, dialogue_text))

    edges = {}

    # helper to add edge
    def add_edge(a, b, dialogue_text=None, ctx=None):
        if a == b:
            key = (a, b)
        else:
            key = tuple(sorted([a, b]))
        entry = edges.setdefault(key, {"count": 0, "dialogs": [], "contexts": []})
        entry["count"] += 1
        if dialogue_text:
            entry["dialogs"].append(dialogue_text)
        if ctx:
            entry["contexts"].append(ctx)

    # use context groups first
    for ctx, spks in context_map.items():
        # if context is non-empty and has multiple speakers, link all pairs
        if ctx.strip():
            # deduplicate preserving order, keep associated dialogue texts
            seen = set()
            unique_spks = []
            for s, text in spks:
                if s not in seen:
                    seen.add(s)
                    unique_spks.append((s, text))
            if len(unique_spks) >= 2:
                for i in range(len(unique_spks)):
                    for j in range(i, len(unique_spks)):
                        a, ta = unique_spks[i]
                        b, tb = unique_spks[j]
                        # include dialogue text when available
                        add_edge(a, b, dialogue_text=ta or tb, ctx=ctx)

    # fallback adjacency linking for contexts that are empty or single-speaker
    seq_spks = [d.get("speaker") for d in dialogues if d.get("speaker")]
    for i in range(len(seq_spks) - 1):
        a = seq_spks[i]
        b = seq_spks[i + 1]
        add_edge(a, b, dialogue_text=None, ctx=None)

    # prepare nodes
    nodes_out = []
    for rep, members in clusters.items():
        canon = canonical[members[0]]
        aliases = []
        nodes_out.append({"id": canon, "members": members, "aliases": aliases, "count": sum(freq.get(m, 0) for m in members)})

    edges_out = []
    for (src, tgt), data in edges.items():
        edges_out.append({"source": src, "target": tgt, "count": data["count"], "dialogs": data["dialogs"], "contexts": data["contexts"]})

    graph = {"nodes": nodes_out, "edges": edges_out}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    if verbose:
        logging.info(f"Wrote dialogue graph to: {output_path}")
        logging.info(f"Nodes: {len(nodes_out)}, Edges: {len(edges_out)}")

    return graph


if __name__ == "__main__":
    try:
        graph = build_dialogue_graph(verbose=True)
        logging.info("dialogue graph built successfully")
    except Exception as e:
        logging.exception("Error building dialogue graph: %s", e)

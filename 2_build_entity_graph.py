import os
import json
import re
import logging
import shutil
from collections import defaultdict, Counter
from difflib import SequenceMatcher, get_close_matches

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
    # remove common punctuation
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

    # exact substring (ex: "john" and "john smith")
    if a in b or b in a:
        return True

    # share last name and at least one has 2+ tokens
    a_tokens = a.split()
    b_tokens = b.split()
    if len(a_tokens) >= 2 and len(b_tokens) >= 1 and a_tokens[-1] == b_tokens[-1]:
        return True
    if len(b_tokens) >= 2 and len(a_tokens) >= 1 and b_tokens[-1] == a_tokens[-1]:
        return True

    # using similarity function
    if similar(a, b) >= 0.86:
        return True

    return False


def build_entity_graph(input_path=None, output_path=None, verbose=True):
    if input_path is None:
        input_path = os.path.join(WORKING_DIR, "entities_relationships.json")
    if output_path is None:
        output_path = os.path.join(WORKING_DIR, "entity_graph.json")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Remove existing output file to ensure we always write a fresh graph
    try:
        if os.path.exists(output_path):
            os.remove(output_path)
            if verbose:
                logging.info(f"Removed existing output file: {output_path}")
    except Exception:
        # non-fatal, continue
        logging.debug("Could not remove existing output file (continuing)")

    with open(input_path, "r", encoding="utf-8") as f:
        rels = json.load(f)

    # collect all unique entity names and frequencies
    entities = []
    freq = Counter()
    for r in rels:
        e1 = r.get("entity1")
        e2 = r.get("entity2")
        if e1:
            entities.append(e1)
            freq[e1] += 1
        if e2:
            entities.append(e2)
            freq[e2] += 1

    uniques = list(dict.fromkeys(entities))  # preserve order
    if verbose:
        logging.info(f"Found {len(entities)} entity mentions, {len(uniques)} unique surface forms")

    # create normalized forms mapping
    norm = {u: normalize_name(u) for u in uniques}

    # prepare DSU
    dsu = DSU(uniques)

    # combine / cluster the same entities (ex: john, john smith, mr. john smith -> cluster 1)
    n = len(uniques)
    for i in range(n):
        for j in range(i + 1, n):
            a = uniques[i]
            b = uniques[j]
            na = norm[a]
            nb = norm[b]
            if are_synonyms(na, nb):
                dsu.union(a, b)

    # build clusters
    clusters = defaultdict(list)
    for u in uniques:
        clusters[dsu.find(u)].append(u)

    # choose canonical name for each cluster: most frequent, tie -> longest
    canonical = {}
    for rep, members in clusters.items():
        members_sorted = sorted(members, key=lambda x: (-freq[x], -len(x)))
        canon = members_sorted[0]
        for m in members:
            canonical[m] = canon

    # build aggregated edges
    edges = {}
    for r in rels:
        a = r.get("entity1")
        b = r.get("entity2")
        if not a or not b:
            continue
        ca = canonical.get(a, a)
        cb = canonical.get(b, b)
        if ca == cb:
            # self-link, ignore or count as internal
            key = (ca, cb)
        else:
            # order to make undirected-like
            key = tuple(sorted([ca, cb]))

        entry = edges.setdefault(key, {"count": 0, "relationships": Counter(), "contexts": []})
        entry["count"] += 1
        rel_type = r.get("relationship") or "interacted"
        entry["relationships"][rel_type] += 1
        ctx = r.get("context")
        if ctx:
            entry["contexts"].append(ctx)

    # prepare output structure
    nodes_out = []
    for rep, members in clusters.items():
        canon = canonical[members[0]]
        aliases = sorted(set(members) - {canon})
        nodes_out.append({"id": canon, "members": members, "aliases": aliases, "count": sum(freq[m] for m in members)})

    edges_out = []
    for (src, tgt), data in edges.items():
        edges_out.append({
            "source": src,
            "target": tgt,
            "count": data["count"],
            "relationships": dict(data["relationships"]),
            "contexts": data["contexts"]
        })

    graph = {"nodes": nodes_out, "edges": edges_out}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    if verbose:
        logging.info(f"Wrote aggregated graph to: {output_path}")
        logging.info(f"Nodes: {len(nodes_out)}, Edges: {len(edges_out)}")

    return graph


if __name__ == "__main__":
    logging.info(f"graph is in the making...")
    try:
        graph = build_entity_graph(verbose=True)
        logging.info(f"graph built successfully")
    except Exception as e:
        # use exception logging to avoid formatting errors and include traceback
        logging.exception("Error building entity graph: %s", e)

"""
Audit layer. Renders proof trees produced by Provenance into human-readable
explanations. Takes the nested node-branch structures from Prove and the
backward reachability matrix from Closure, and formats them as annotated
reasoning chains with per-step confidence scores. Separating this from
Provenance keeps proof construction and proof presentation as distinct
concerns — Provenance deals in tensors and logic, Audit deals in names
and strings.
"""

import numpy as np
from .tree import Tree

def extract_path(proof):
    """
    Flatten a proof tree into a linear path of entity indices.

    Folds the tree using Tree.fold, concatenating and deduplicating
    adjacent indices at each Pair node. Returns the first valid path
    found among any branches. Returns None if no path can be extracted.
    """
    def folder(tag, *args):
        if tag == 'leaf':
            # Leaf integers stay as-is
            return args[0]
        elif tag == 'pair':
            fst, snd = args
            # Base case: both are integers
            if isinstance(fst, (int, np.integer)) and isinstance(snd, (int, np.integer)):
                return [int(fst), int(snd)]
            # Recursive case: both are paths (lists)
            if isinstance(fst, list) and isinstance(snd, list):
                # Concatenate and deduplicate
                path = fst + snd[1:]
                return [path[i] for i in range(len(path)) if i == 0 or path[i] != path[i-1]]
            return None
        elif tag == 'dict':
            folded_dict = args[0]
            if 'branches' in folded_dict:
                # branches are already folded - pick first valid one
                paths = [p for p in folded_dict['branches'] if p is not None]
                return paths[0] if paths else None
            return None
        elif tag == 'list':
            # Return folded list items as-is
            return args[0]
        elif tag == 'tuple':
            # Tuples in proof dicts are just metadata, ignore
            return None
        return None
    return Tree.fold(folder, proof, None)


def format_branch(path, backward, names, relation="is related to"):
    """
    Format a single reasoning path as an annotated string.

    Walks consecutive pairs in path, annotating each step with the
    edge strength from backward and the relation label. Returns None
    if path is None.
    """
    if path is None:
        return None
    parts = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        parts.append(f"{names[u]} ({backward[u, v]:.3f}) {relation}")
    parts.append(names[path[-1]])
    return " → ".join(parts)


def format_proof(proof, backward, names, relation="is related to"):
    """
    Format a full proof tree into a human-readable explanation.

    For a branching proof (dict), extracts and formats each branch path,
    deduplicates them, and returns a dict with a label and branch list.
    For a leaf proof (Tree.Pair), extracts and formats the single path.
    Returns "No proof found" if the proof is None or yields no valid paths.
    """
    if proof is None:
        return "No proof found"
    if isinstance(proof, dict):
        u, v = proof['node']
        label = f"{names[u]} {relation} {names[v]} because:"
        paths = [extract_path(b) for b in proof['branches']]
        paths = [p for p in paths if p is not None]
        if not paths:
            return "No proof found"
        branches = [format_branch(p, backward, names, relation) for p in paths]
        branches = list(dict.fromkeys(b for b in branches if b is not None))
        return {'node': label, 'branches': branches}
    else:
        path = extract_path(proof)
        if path is None:
            return "No proof found"
        return format_branch(path, backward, names)
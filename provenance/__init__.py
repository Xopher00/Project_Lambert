"""
Provenance package. Proof extraction and formatting over relational tensors.

Contains three modules:

- tree: structural recursion utilities (Tree, Tree.Pair, fold, map, zip)
- audit: proof formatting (extract_path, format_branch, format_proof)
- provenance: proof extraction (Provenance, Witnesses, Prove, Query)

# Under review. As lattice traversal matures in CategoryExplorer, explicit
# witness-based proof trees may become redundant. These modules are retained
# as the theory and implementation are sound, but their role in the broader
# architecture is being reassessed.
"""

from provenance.tree import Tree
from provenance.audit import extract_path, format_branch, format_proof
from provenance.provenance import Provenance

"""
Provenance package. Proof extraction and formatting over relational tensors.
Retired to legacy. The active codebase no longer uses this package.

The witness-based proof tree approach documented here has been superseded by
lattice traversal paths through the concept hierarchy constructed by
CategoryExplorer. Where provenance reconstructed proof trees by tracking
intermediate Join witnesses, the concept lattice provides the same information
structurally: formal concept intents are the reason set for a query result,
and lattice navigation (meet, join, containment) gives the reasoning path.

See research/lattice/explorer.md "What the explorer does not do" for the
current theoretical framing of provenance in the lattice model.

Modules retained here as historical record:
- tree: structural recursion utilities (Tree, Tree.Pair, fold, map, zip)
- audit: proof formatting (extract_path, format_branch, format_proof)
- provenance: proof extraction (Provenance, Witnesses, Prove, Query)

The theory and implementation are sound. The architecture decision to retire
this layer was made when the lattice embedding became the primary provenance
mechanism and explicit witness storage became redundant.
"""

from legacy.provenance.tree import Tree
from legacy.provenance.audit import extract_path, format_branch, format_proof
from legacy.provenance.provenance import Provenance

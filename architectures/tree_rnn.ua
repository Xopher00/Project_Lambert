# tree_rnn.ua
# Recursive Neural Network: endofunctor A + S^2 (Gavranovic I.2, J.3)
# Algebra = catamorphism = fold over binary tree.
#
# leaf_node: A component — embed one token A into representation S.
# inner_node: S x S component — combine two children's representations into S.
#
# leaf: embed(a) -> s  (morphism-derived: x=None (no child), y=payload[0]=token)
# node: combine(s_left, s_right) -> s  (custom cell handles both children)
# Same weights P at every node (weight tying).

semiring std:
    plus  = sum
    times = multiply
    zero  = 0.0
    one   = 1.0

sort repr, token

arch TreeRNN:
    cases:
        leaf_node:  leaf        data=1  cell=ops.leaf_cell
        inner_node: recursive=2 data=0  cell=ops.inner_cell

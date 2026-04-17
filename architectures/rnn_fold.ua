# rnn_fold.ua
# Folding RNN: endofunctor 1 + A x S (Gavranovic I.1, J.1)
# Algebra = catamorphism = fold list of inputs into hidden state.
#
# base: 1 component — no input, no previous state — produces initial hidden state.
# step: A x S component — input token (data) + previous hidden (recursive) -> new hidden.
#
# cell(h, a) -> h: same weights P at every step (weight tying).
# Morphism-derived cell: x = child_results[0] (prev_hidden), y = payload[0] (W_h weight matrix).

semiring std:
    plus  = sum
    times = multiply
    zero  = 0.0
    one   = 1.0

sort hidden

morphism recur    : hidden -> hidden  via "si,ij->sj"
morphism activate : hidden -> hidden  via "sh->sh"    op ops.activate  arity unary

path rnn = recur activate

arch FoldingRNN:
    cases:
        base:     leaf  data=0  cell=ops.init_hidden
        rnn_step: node  data=1  morphisms=rnn

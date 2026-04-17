# rnn_stream.ua
# Unfolding RNN / Stream generator: endofunctor O x S (Gavranovic I.3, H.4)
# Coalgebra = anamorphism = unfold hidden state into output stream.
#
# At each step: state S -> output O (read) + next state S (transition).
# No external input — the network generates freely from its state.
# Same weights P at every step (weight tying).
#
# enter: pass-through (state is the primary signal; token slot unused)
# compute (via morphisms): hidden -> new_hidden (state transition via semiring contract)
# emit: hidden -> output O

semiring std:
    plus  = sum
    times = multiply
    zero  = 0.0
    one   = 1.0

sort hidden, output

morphism pass_through : hidden -> hidden  via "sh->sh"     op ops.pass_through  arity unary
morphism transition   : hidden -> hidden  via "si,ij->sj"
morphism activate     : hidden -> hidden  via "sh->sh"     op ops.activate      arity unary
morphism read_output  : hidden -> output  via "sh,ho->so"  op ops.read_output   arity unary

path step_path = transition activate

arch StreamRNN:
    cases:
        stream_step: node  data=1  output=1  morphisms=step_path
    step:
        enter = pass_through
        emit  = read_output

# mealy.ua
# Mealy Machine / Full RNN: endofunctor I -> O x S (Gavranovic I.4, Example 2.11)
# Coalgebra: cell^Mealy: P x S x I -> O x S
# At each step: state + input -> output + new state.
# Same weights P at every step (weight tying).
#
# Distinguished from StreamRNN: output depends on BOTH current state AND current input.
# step: enter = embed_input  (token -> hidden: inject input into state space)
#       emit  = emit_output  (hidden -> output: project to output space)

semiring std:
    plus  = sum
    times = multiply
    zero  = 0.0
    one   = 1.0

sort hidden, input, output

morphism embed_input  : input -> hidden  via "sd,dh->sh"  op ops.embed_input  arity unary
morphism rnn_step     : hidden -> hidden  via "si,ij->sj"
morphism activate     : hidden -> hidden  via "sh->sh"     op ops.activate     arity unary
morphism emit_output  : hidden -> output  via "sh,ho->so"  op ops.emit_output  arity unary

path compute = rnn_step activate

arch MealyRNN:
    cases:
        mealy_step: node  data=1  output=1  morphisms=compute
    step:
        enter = embed_input
        emit  = emit_output

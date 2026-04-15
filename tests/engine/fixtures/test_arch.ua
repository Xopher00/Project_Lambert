semiring attn:
    contract = ops.join_fn
    compiler = ops.compile_einsum

sort model, q, scores, mixed

morphism q_proj : model -> q via "sd,dh->sh"
morphism score  : q -> scores via "sh,sh->s"
morphism mix    : scores -> mixed via "s,sh->sh"
morphism out    : mixed -> model via "sh,hd->sd"

path read = q_proj score mix out
path attn = q_proj score mix out  residual

fan kv = q_proj & score  merge dict

arch Transformer:
    algebra:
        case input: recursive=0  data=1  cell=identity
        case block: recursive=1  data=1  morphisms = read

from torch_semiring_einsum import compile_equation

from engine.runtime import MorphismSpec, compile_morphism, chain

from streamlined.relational_einsum import join_einsum_forward, residuate_einsum_forward

_SPECS = [
    MorphismSpec("realize",   join_einsum_forward,      "j,ji->i", "j", "i", compile_equation, lambda x, y: (x, y.T)),
    MorphismSpec("propagate", join_einsum_forward,      "i,ij->j", "i", "j", compile_equation, lambda x, y: (x, y)),
    MorphismSpec("abstract",  residuate_einsum_forward, "ij,i->j", "i", "j", compile_equation, lambda x, y: (y, x)),
    MorphismSpec("support",   residuate_einsum_forward, "ji,j->i", "j", "i", compile_equation, lambda x, y: (y.T, x)),
]

_compiled = {s.name: compile_morphism(s) for s in _SPECS}


class Compose:
    """
    Relational compositions built on top of the path engine and user defined einsum methods.
    """

    def __init__(self):
        self.Scores  = _compiled["realize"]
        self.Hop     = _compiled["propagate"]
        self.Attend  = chain([_compiled["realize"], _compiled["propagate"]])
        self.Recall  = chain([_compiled["abstract"], _compiled["support"]])
        self.Correct = chain([_compiled["support"], _compiled["propagate"]])
        self.Project = chain([_compiled["propagate"], _compiled["realize"]])

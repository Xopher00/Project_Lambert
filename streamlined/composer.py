from torch_semiring_einsum import compile_equation

from engine.path_engine import PathEngine, LegSpec
from streamlined.relational_einsum import join_einsum_forward, residuate_einsum_forward

_LEGS = [
    LegSpec("realize",   join_einsum_forward,      "j,ji->i", "j", "i", compile_equation, lambda x, y: (x, y.T)),
    LegSpec("propagate", join_einsum_forward,      "i,ij->j", "i", "j", compile_equation, lambda x, y: (x, y)),
    LegSpec("abstract",  residuate_einsum_forward, "ij,i->j", "i", "j", compile_equation, lambda x, y: (y, x)),
    LegSpec("support",   residuate_einsum_forward, "ji,j->i", "j", "i", compile_equation, lambda x, y: (y.T, x)),
]

class Compose:
    """
    Relational compositions built on top of the path engine and user defined einsum methods.
    """

    def __init__(self):
        self.coder = PathEngine(_LEGS)

        self.Scores = self.coder.op("realize")
        self.Hop = self.coder.op("propagate")
        self.Attend = self.coder.op("realize propagate")
        self.Recall = self.coder.op("abstract support")
        self.Correct = self.coder.op("support propagate")
        self.Project = self.coder.op("propagate realize")    
 

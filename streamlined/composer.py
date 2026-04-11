from streamlined.path_engine import PathEngine
from streamlined.relational_einsum import join_einsum_forward, residuate_einsum_forward

class Compose:
    """
    Relational compositions built on top of the path engine and user defined einsum methods.
    """

    def __init__(self):
        self.coder = PathEngine({
            "realize":   (join_einsum_forward,      "j,ji->i", lambda x, y: (x, y.T)),
            "propagate": (join_einsum_forward,      "i,ij->j", lambda x, y: (x, y)),
            "abstract":  (residuate_einsum_forward, "ij,i->j", lambda x, y: (y, x)),
            "support":   (residuate_einsum_forward, "ji,j->i", lambda x, y: (y.T, x)),
        })

        self.Scores = self.coder.op("realize")
        self.Hop = self.coder.op("propagate")
        self.Attend = self.coder.op("realize propagate")
        self.Recall = self.coder.op("abstract support")
        self.Correct = self.coder.op("support propagate")
        self.Project = self.coder.op("propagate realize")    
 

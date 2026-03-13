from fixpoint import FixpointIterator
# from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from algebra import *
from embed import Embed

""""
This layer sits on top of our attention / transformer layer.
This layer is inspired by category theory. a category is defined as a relationship
new categories can be defined by relationship compositions
an entity can be in many categories. 
some categories are isomorphic. even tho they contain different entities, or their intents are different,
the structure of the relations is the same. we have a method for finding these categories
After initializing an instance of the multi head attention class and embedding data in it,
pass that instance into category explorer. this will explore every single possible relationship between the entities in the data
even relatively small datasets can reveal a lot of information that was not explicitly there
This is an advanced form of transitive closure
"""

class CategoryExplorer(Embed):
    def __init__(self, mha, eps=1e-3):
        super().__init__() 
        self.mha = mha
        self.eps = eps
        self.categories = {}

    # def _intent_key(self):
    #     parts = []
    #     for name, (intent, _) in self.mha.intents.items():
    #         parts.append(tuple((intent / self.eps).astype(int)))
    #     return tuple(parts)
    
    def _intent_key(self):
        extent = self.mha.fp.state
        return tuple((extent / self.eps).astype(int))
       
    # Perform an initial exploration 
    def explore(self, n_entities):
        covered = set()
        for i in range(n_entities):
            if i in covered:
                continue
            if i % 50 == 0:
                print(f"  exploring entity {i}/{n_entities}  "
                    f"categories={len(self.categories)}  "
                    f"covered={len(covered)}")
            self.mha.intents = {} 
            hits, intents = self.mha.retrieve([i])
            if len(hits) == 0:
                continue
            key = self._intent_key()
            extent = self.mha.fp.state.copy()
            if key not in self.categories:
                self.categories[key] = {
                    'intents':  dict(self.mha.intents),
                    'extent':   extent,
                }
                covered.update(np.flatnonzero(extent > self.eps).tolist())
        return self.categories

    def _concept_fixpoint(self, R, seed, temp, max_iters=20, eps=1e-3):
        active = np.flatnonzero(seed > eps)
        if len(active) == 0:
            return seed        
        def _f(state, temp):
            hits, _ = self.mha.retrieve(np.flatnonzero(state > eps).tolist())
            return self.mha.fp.state.copy(), None         
        fp = FixpointIterator(
            f         = _f,
            state0    = seed.copy(),
            eps       = eps,
            max_iters = max_iters,
        )
        return fp.run()
    
    def explore_lattice(self, n_entities):
        print('Performing initial concept exploration . . .')
        self.explore(n_entities)
        print('Initial exploration phase complete.')
        emb_new = np.zeros((n_entities, len(self.categories)))
        print('Beginning full concept exploration . . .')
        for col, (key, cat) in enumerate(self.categories.items()):
            emb_new[:, col] = np.where(cat['extent'] > self.eps, cat['extent'], 0)
        emb, EmbR, rep_cols = self.ConceptEmbed(emb_new, temp=self.mha.heads[0].fp.temp)
        print('Transitive closure reached.')
        return emb, EmbR, rep_cols

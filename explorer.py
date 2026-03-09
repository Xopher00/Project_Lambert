from fixpoint import FixpointIterator
from numpy import np
from algebra import *

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

class CategoryExplorer:
    def __init__(self, mha, eps=1e-3):
        self.mha = mha
        self.eps = eps
        self.categories = {}

    def _intent_key(self):
        parts = []
        for name, (intent, _) in self.mha.intents.items():
            parts.append(tuple((intent / self.eps).astype(int)))
        return tuple(parts)
    
    # select a nonzero index from the extent vector of a category
    def _representative(self, extent):
        nonzero = np.where(extent > self.eps)[0]
        if len(nonzero) == 0:
            return None
        return nonzero[np.argmin(extent[nonzero])]
    
    # check if a relationship between two extents has already been discovered
    def _co_occurs(self, i, j):
        return any(
            cat['extent'][i] > self.eps and cat['extent'][j] > self.eps
            for cat in self.categories.values()
        )

    # Perform an initial exploration 
    def explore(self, n_entities):
        for i in range(n_entities):
            hits, intents = self.mha.retrieve([i])
            if len(hits) == 0:
                continue
            key = self._intent_key()
            if key not in self.categories:
                self.categories[key] = {
                    'intents': dict(self.mha.intents), 
                    'extent': self.mha.fp.state.copy()
                }
        return self.categories
    
    def isomorphic_groups(self):
        groups = {}
        for key, cat in self.categories.items():
            iso_key = tuple(
                tuple(sorted(intent[intent > self.eps], reverse=True))
                for _, (intent, _) in cat['intents'].items()
            )
            if iso_key not in groups:
                groups[iso_key] = []
            groups[iso_key].append(key)
        return groups
    
    def prune(self):
        """Drop singletons and collapse isomorphic groups."""
        # Drop singletons — categories with only one entity in the extent
        self.categories = {
            k: v for k, v in self.categories.items()
            if np.sum(v['extent'] > self.eps) > 1
        }

        # Collapse isomorphic groups — union extents, keep one representative
        groups = self.isomorphic_groups()
        collapsed = {}
        for iso_key, members in groups.items():
            # Union of all extents across isomorphic members
            union_extent = np.maximum.reduce([
                self.categories[k]['extent'] for k in members
            ])
            # Keep the first member's intents as representative
            representative = self.categories[members[0]]
            collapsed[members[0]] = {
                'intents': representative['intents'],
                'extent': union_extent
            }
        self.categories = collapsed
    
    def _lattice_step(self, state, temp):
        keys = list(self.categories.keys())
        for a, key_a in enumerate(keys):
            for key_b in keys[a+1:]:
                i = self._representative(self.categories[key_a]['extent'])
                j = self._representative(self.categories[key_b]['extent'])
                if i is None or j is None:
                    continue
                if self._co_occurs(i, j):
                    continue
                hits, _ = self.mha.retrieve([i, j])
                if len(hits) == 0:
                    continue
                new_key = self._intent_key()
                if new_key not in self.categories:
                    self.categories[new_key] = {'intents': dict(self.mha.intents), 'extent': self.mha.fp.state.copy()}
        return np.maximum.reduce([cat['extent'] for cat in self.categories.values()]), None
    
    def explore_lattice(self, n_entities):
        self.explore(n_entities)
        union = np.maximum.reduce([cat['extent'] for cat in self.categories.values()])
        fp = FixpointIterator(
            f         = self._lattice_step,
            energy_fn = lambda new, old, aux: float(Sum(Abs(new - old))),
            state0    = union,
            eps       = self.eps,
            max_iters = 50,
        )
        fp.run()
        # Prune categories with empty intents - where no relationship was found
        self.categories = {
            k: v for k, v in self.categories.items() 
            if v['intents'] and any(np.any(intent > self.eps) for _, (intent, _) in v['intents'].items())
        }
        self.prune()
        print(f"Valid categories after pruning: {len(self.categories)}")
        return self.categories

import numpy as np
from typing import Optional
from dataclasses import dataclass, field
from algebra import *
from embed import Embed
from language import Labeler
from attention import Attention, MultiHeadAttention
from explorer import CategoryExplorer

@dataclass
class Lambert:

    # --- config (set at construction) ---
    entity_labels: list
    embed_temp:    float = 1.0
    attn_temp:     float = 1.0
    eps:           float = 1e-3
    label:         bool  = False

    # --- results (populated by run()) ---
    # heads[name] = {'emb': ndarray, 'rep_cols': list, 'feature_labels': list}
    heads:         Optional[dict] = field(default=None, repr=False)
    # concept_space = {'emb': ndarray, 'unique': ndarray, 'inverse': ndarray, 'categories': dict}
    concept_space: Optional[dict] = field(default=None, repr=False)
    # iso_index = {'exact': {id: {'signature': tuple}}, 'near': {id: {'signatures': list}}, 'matrix': ndarray}
    iso_index:     Optional[dict] = field(default=None, repr=False)
    # labels = {'categories': dict, 'meta': dict, 'iso': dict}
    labels:        Optional[dict] = field(default=None, repr=False)
    # --- internal components ---
    explorer:      Optional[object] = field(default=None, repr=False)

    def _get_embeddings(self, relations: dict) -> dict:
        embed = Embed()
        self.heads = {}
        for name, (R, feature_labels) in relations.items():
            emb, _, rep_cols = embed.ConceptEmbed(R, temp=self.embed_temp, eps=self.eps)
            self.heads[name] = {
                'emb':            emb,
                'rep_cols':       rep_cols,
                'feature_labels': feature_labels
            }
        return {name: v['emb'] for name, v in self.heads.items()}
        
    def _build_mha(self) -> MultiHeadAttention:
        attn_heads = []
        names = []
        for name, h in self.heads.items():
            attn_heads.append(Attention(h['emb'], temp=self.attn_temp, eps=self.eps))
            names.append(name)
        return MultiHeadAttention(heads=attn_heads, names=names, eps=self.eps)
       
    def _explore(self, n_entities: int):
        mha = self._build_mha()
        self.explorer = CategoryExplorer(mha, eps=self.eps)
        emb, _, rep_cols = self.explorer.explore_lattice(n_entities)
        unique, inverse = np.unique(emb, axis=0, return_inverse=True)
        self.concept_space = {
            'emb':        emb,
            'rep_cols':   rep_cols,
            'unique':     unique,
            'inverse':    inverse,
            'categories': self.explorer.categories
        }

    def _isomorphic(self):
        unique  = self.concept_space['unique']
        inverse = self.concept_space['inverse']

        coeffs = sorted(set(unique[unique > self.eps].tolist()))
        matrix = np.zeros((len(unique), len(coeffs)))
        for j, c in enumerate(coeffs):
            matrix[:, j] = (unique == c).any(axis=1).astype(float)

        self.iso_index = {
            'coefficients': coeffs,
            'matrix':       matrix[inverse]
        }

    def run(self, relations: dict, n_entities: int) -> tuple:
        print('getting embeddings...')
        self._get_embeddings(relations)
        print(f'embeddings ready: {list(self.heads.keys())}')

        print('exploring lattice...')
        self._explore(n_entities)
        print(f'exploration complete: emb shape={self.concept_space["emb"].shape}, {len(self.concept_space["categories"])} categories')

        print('finding isomorphic relationships...')
        self._isomorphic()
        print(f'iso index: {len(self.iso_index["coefficients"])} coefficients')

        if self.label:
            print('labeling discovered categories...')
            labeler = Labeler(
                categories=self.concept_space['categories'],
                emb_cat=self.concept_space['emb'],
                feature_labels={n: h['feature_labels'] for n, h in self.heads.items()},
                rep_cols={n: h['rep_cols'] for n, h in self.heads.items()},
                cat_rep_cols=self.concept_space['rep_cols'],
                entity_labels=self.entity_labels,
                eps=self.eps
            )
            labeler.run()
            self.labels = {
                'categories': labeler.labels,
                'meta':       labeler.meta_labels,
                'iso':        labeler.iso_labels
            }

        print(f'categories: {len(self.concept_space["categories"])}')
        print(f'iso coefficients: {len(self.iso_index["coefficients"])}')
        
import numpy as np
from typing import Optional
from dataclasses import dataclass, field
import asyncio, aiohttp, json, os, nest_asyncio

from embed import Embed
from language import Labeler
from provenance import Provenance
from attention import Attention, MultiHeadAttention
from explorer import CategoryExplorer

@dataclass
class Lambert:

    # --- config (set at construction) ---
    entity_labels: list
    embed_temp:  float = 0.1
    attn_temp:   float = 0.9
    eps:         float = 1e-3
    label:       bool  = False

    # --- results (populated by run()) ---
    emb_cat:    Optional[np.ndarray] = field(default=None, repr=False)
    EmbR:       Optional[np.ndarray] = field(default=None, repr=False)
    rep_cols:   Optional[list]       = field(default=None, repr=False)
    categories: Optional[dict]       = field(default=None, repr=False)
    isos:       Optional[dict]       = field(default=None, repr=False)
    labels:     Optional[dict]       = field(default=None, repr=False)
    meta_labels: Optional[dict] = field(default=None, repr=False)
    iso_labels:  Optional[dict] = field(default=None, repr=False)

    # --- internal components ---
    explorer:   Optional[object]     = field(default=None, repr=False)

    def _get_embeddings(self, relations: dict) -> dict:
        embed = Embed()
        embeddings = {}
        self.feature_labels = {}
        self.rep_cols = {}
        for name, (R, labels) in relations.items():
            emb, _, rep_cols = embed.ConceptEmbed(R, temp=self.embed_temp, eps=self.eps)
            embeddings[name] = emb
            self.feature_labels[name] = labels
            self.rep_cols[name] = rep_cols
        return embeddings
    
    def _build_mha(self, embeddings: dict) -> MultiHeadAttention:
        heads = []
        names = []
        for name, emb in embeddings.items():
            heads.append(Attention(emb, temp=self.attn_temp, eps=self.eps))
            names.append(name)
        return MultiHeadAttention(heads=heads, names=names, eps=self.eps)
    
    def _explore(self, mha: MultiHeadAttention, n_entities: int):
        self.explorer = CategoryExplorer(mha, eps=self.eps)
        emb_cat, _, rep_cols = self.explorer.explore_lattice(n_entities)
        self.emb_cat = emb_cat
        self.cat_rep_cols = rep_cols
        self.categories = self.explorer.categories

    def _detect_isos(self) -> dict:
        pass

    def run(self, relations: dict, n_entities: int) -> tuple:
        print('getting embeddings...')
        embeddings = self._get_embeddings(relations)
        print(f'embeddings ready: {list(embeddings.keys())}')
        
        print('building MHA...')
        mha = self._build_mha(embeddings)
        print(f'MHA ready: {len(mha.heads)} heads')
        
        print('exploring lattice...')
        self._explore(mha, n_entities)
        print(f'exploration complete: emb_cat shape={self.emb_cat.shape}, {len(self.categories)} categories')

        print('Labeling discovered categories...')
        if self.label:
            labeler = Labeler(
                categories=self.categories,
                emb_cat=self.emb_cat,
                feature_labels=self.feature_labels,
                rep_cols=self.rep_cols,
                eps=self.eps
            )
            labeler.run()
            self.labels = labeler.labels
            self.meta_labels = labeler.meta_labels
            self.iso_labels = labeler.iso_labels
            self.isos = labeler.isos
            print(f'Base categories: {len(self.labels)}')
            print(f'Meta categories: {len(self.meta_labels)}')
            print(f'Isomorphic groups: {len(self.isos)}')
        
        return self.emb_cat, self.categories, self.rep_cols
    

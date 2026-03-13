import numpy as np
from typing import Optional
from dataclasses import dataclass, field
from algebra import *
from embed import Embed
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

    def _chunk(self, R, vocab):
        embed = Embed()
        binary = (R > 0).astype(float)
        sim = embed.GramMatrix(binary.T, temp=1.0)  # (n_features, n_features)
        chunk_size = min(int(np.sqrt(R.shape[1])), R.shape[0] - 1)
        n_chunks = max(2, R.shape[1] // chunk_size)
        soft = embed.SoftMax(sim, temp=1.0, axis=1)
        soft   = embed.SoftMax(sim, temp=1.0, axis=1)
        order  = np.argsort(np.argmax(soft, axis=1))  # sort features by their soft assignment
        labels = np.zeros(R.shape[1], dtype=int)
        labels[order] = np.arange(R.shape[1]) * n_chunks // R.shape[1]
        return {f'head_{i}': (R[:, np.where(labels==i)[0]], 
                [vocab[j] for j in np.where(labels==i)[0]]) for i in range(n_chunks)}

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

    def _map_values(self):
        emb_vals = set(float(v) for v in self.concept_space['emb'].flat if v > self.eps)
        return {
            float(v): f"[{head_name}] {self.heads[head_name]['feature_labels'][self.heads[head_name]['rep_cols'][j]]}"
            for cat in self.concept_space['categories'].values()
            for head_name, (intent_vec, _) in cat['intents'].items()
            for j, v in enumerate(intent_vec)
            if float(v) in emb_vals
        }

    def run(self, relations: dict = None, n_entities: int = None, R: np.ndarray = None, vocab: list = None) -> tuple:

        if R is not None:
            print('Chunking . . . ')
            relations = self._chunk(R, vocab)
            n_entities = R.shape[0]
            print(f'Partitioned data into {len(relations)} chunks.')

        print('getting embeddings...')
        self._get_embeddings(relations)
        print(f'embeddings ready: {list(self.heads.keys())}')

        print('exploring lattice...')
        self._explore(n_entities)
        print(f'exploration complete: emb shape={self.concept_space["emb"].shape}, {len(self.concept_space["categories"])} categories')

        print(f'Identifying strongest defining traits . . . ')
        self.concept_space['feature_map'] = self._map_values()

        print(f'categories: {len(self.concept_space["categories"])}')
        
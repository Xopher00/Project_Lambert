"""
Top-level model. Lambert is the main entry point for the full pipeline:
chunking, embedding, lattice exploration, and feature mapping.

Given a set of relation matrices (one per head, or a single wide matrix
to be partitioned automatically), Lambert builds concept embeddings for
each head, assembles a MultiHeadAttention, explores the full concept
lattice via CategoryExplorer, and constructs a human-readable lookup
table mapping extent keys to the feature labels that caused entities to
be grouped together.

All results are stored on the instance after run() completes and can be
accessed via concept_space, heads, and labels.
"""

import numpy as np
from query import Query
from typing import Optional
from dataclasses import dataclass, field
from lattice import Embed, Attention, MultiHeadAttention, CategoryExplorer


@dataclass
class Lambert:
    """
    Full pipeline from relation matrices to a closed concept lattice.

    Constructed with configuration parameters, then driven by a single
    call to run(). Results are stored on the instance in concept_space,
    heads, and labels.

    Parameters
    ----------
    entity_labels : list
        Names of the entities in the relation matrices.
    embed_temp : float, optional
        Temperature used during concept embedding. Default is 1.0.
    attn_temp : float, optional
        Temperature used during attention retrieval. Default is 1.0.
    eps : float, optional
        Convergence threshold and extent membership threshold. Default is 1e-3.
    label : bool, optional
        Vestigial. Originally a flag to trigger LLM-based category labelling,
        which has since been removed. Default is False.

    Attributes
    ----------
    heads : dict or None
        Populated by run(). Maps head name to a dict with keys:

        - ``'emb'``: the concept embedding matrix for this head
        - ``'rep_cols'``: representative column indices from the relation matrix
        - ``'feature_labels'``: human-readable labels for each feature column
    concept_space : dict or None
        Populated by run(). Contains:

        - ``'emb'``: the final embedding over the closed concept lattice
        - ``'rep_cols'``: representative concept indices
        - ``'unique'``: deduplicated embedding rows, for extent key lookup
        - ``'inverse'``: maps each entity back to its row in unique
        - ``'categories'``: the full category dict from CategoryExplorer
        - ``'feature_map'``: human-readable lookup from extent key to feature label
    iso_index : dict or None
        Vestigial. Previously used to index structurally isomorphic categories.
    labels : dict or None
        Reserved for downstream labelling. Not yet populated.
    explorer : CategoryExplorer or None
        The CategoryExplorer instance used during run(). Retained for inspection.
    """

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
        """
        Partition a wide relation matrix into chunks for multi-head embedding.

        Computes feature-feature similarity via GramMatrix, assigns each feature
        to a chunk using SoftMax and argmax, and splits R by chunk assignment.
        Features that tend to co-occur on the same entities are grouped together,
        so each resulting chunk is a coherent subset of the feature space.

        This is a preprocessing step for efficiency: partitioning a wide matrix
        before embedding reduces the cost of ConceptEmbed on each head.

        Parameters
        ----------
        R : ndarray, shape (n_entities, n_features)
            The relation matrix to partition.
        vocab : list of str
            Feature labels corresponding to columns of R.

        Returns
        -------
        dict
            Maps chunk name (``'head_0'``, ``'head_1'``, ...) to a tuple of:

            - ``ndarray``: the sub-matrix of R for this chunk
            - ``list of str``: the feature labels for this chunk's columns
        """
        embed = Embed()
        binary = embed.Relu(R).astype(float)
        sim = embed.GramMatrix(binary.T, temp=self.embed_temp)  # (n_features, n_features)
        chunk_size = min(int(np.sqrt(R.shape[1])), R.shape[0] - 1)
        n_chunks = max(2, R.shape[1] // chunk_size)
        soft = embed.SoftMax(sim, temp=self.embed_temp, axis=1)
        order  = np.argsort(np.argmax(soft, axis=1))  # sort features by their soft assignment
        labels = np.zeros(R.shape[1], dtype=int)
        labels[order] = np.arange(R.shape[1]) * n_chunks // R.shape[1]
        return {f'head_{i}': (R[:, np.where(labels==i)[0]],
                [vocab[j] for j in np.where(labels==i)[0]]) for i in range(n_chunks)}

    def _get_embeddings(self, relations: dict) -> dict:
        """
        Compute concept embeddings for each head.

        Runs ConceptEmbed on each relation matrix and stores the resulting
        embedding, representative column indices, and feature labels in
        self.heads.

        Parameters
        ----------
        relations : dict
            Maps head name to a tuple of (R, feature_labels), where R is a
            relation matrix and feature_labels is a list of column names.

        Returns
        -------
        dict
            Maps head name to its embedding matrix.
        """
        embed = Embed()
        self.heads = {}
        for name, (R, feature_labels) in relations.items():
            emb, EmbR, rep_cols = embed.ConceptEmbed(R, temp=self.embed_temp, eps=self.eps)
            self.heads[name] = {
                'emb':            emb,
                'EmbR':           EmbR,
                'rep_cols':       rep_cols,
                'feature_labels': feature_labels
            }
        return {name: v['emb'] for name, v in self.heads.items()}

    def _build_mha(self) -> MultiHeadAttention:
        """
        Assemble a MultiHeadAttention from the computed head embeddings.

        Creates one Attention instance per head using the embedding stored in
        self.heads, then wraps them in a MultiHeadAttention.

        Returns
        -------
        MultiHeadAttention
            The assembled multi-head attention instance, ready for lattice
            exploration.
        """
        attn_heads = []
        names = []
        for name, h in self.heads.items():
            attn_heads.append(Attention(h['emb'], temp=self.attn_temp, eps=self.eps))
            names.append(name)
        return MultiHeadAttention(heads=attn_heads, names=names, eps=self.eps)

    def _explore(self, n_entities: int):
        """
        Build the MultiHeadAttention and run full lattice exploration.

        Constructs the MHA, wraps it in a CategoryExplorer, and calls
        explore_lattice to discover and close the concept lattice. The
        resulting embedding and category dict are stored in self.concept_space.

        Deduplicates embedding rows via np.unique so that entities with
        identical embeddings can be identified as belonging to the same
        category. An entity can appear in more than one category; unique
        and inverse together make it possible to look up which categories
        each entity belongs to.

        Parameters
        ----------
        n_entities : int
            Number of entities to explore.
        """
        mha = self._build_mha()
        self.explorer = CategoryExplorer(mha, eps=self.eps)
        emb, EmbR, rep_cols = self.explorer.explore_lattice(n_entities)
        self.query = Query(self)
        unique, inverse = np.unique(emb, axis=0, return_inverse=True)
        self.concept_space = {
            'emb':        emb,
            'EmbR':       EmbR,
            'rep_cols':   rep_cols,
            'unique':     unique,
            'inverse':    inverse,
            'categories': self.explorer.categories
        }

    def _map_values(self):
        """
        Build a human-readable lookup table from extent keys to feature labels.

        Iterates over all discovered categories and their per-head intents,
        mapping each non-zero intent value to the feature label that produced
        it. The result tells a human reader why entities were grouped together
        — which features the model determined they have in common.

        Extent values are used as dict keys, not as numeric quantities. Each
        value identifies a specific entry in the concept lattice rather than
        expressing a degree of membership.

        Returns
        -------
        dict
            Maps each distinct extent value (float) to the feature label
            string of the form ``'[head_name] feature_name'``.
        """
        emb_vals = set(float(v) for v in self.concept_space['emb'].flat if v > self.eps)
        return {
            float(v): f"[{head_name}] {self.heads[head_name]['feature_labels'][self.heads[head_name]['rep_cols'][j]]}"
            for cat in self.concept_space['categories'].values()
            for head_name, (intent_vec, _) in cat['intents'].items()
            for j, v in enumerate(intent_vec)
            if float(v) in emb_vals
        }

    def run(self, relations: dict = None, n_entities: int = None, R: np.ndarray = None, vocab: list = None) -> tuple:
        """
        Execute the full Lambert pipeline.

        Accepts either a pre-partitioned dict of relation matrices or a single
        wide matrix to be partitioned automatically via _chunk. Runs embedding,
        lattice exploration, and feature mapping in sequence. All results are
        stored on the instance — nothing is returned.

        Parameters
        ----------
        relations : dict, optional
            Pre-partitioned relation matrices. Maps head name to a tuple of
            (R, feature_labels). Required if R is not provided.
        n_entities : int, optional
            Number of entities. Required if R is not provided.
        R : ndarray, shape (n_entities, n_features), optional
            A single wide relation matrix to be partitioned automatically.
            If provided, vocab must also be given.
        vocab : list of str, optional
            Feature labels for the columns of R. Required if R is provided.
        """
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



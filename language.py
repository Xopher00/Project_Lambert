"""
Natural language labelling layer. Uses an LLM to assign human-readable
labels to the categories, meta-categories, and structural groups discovered
by Lambert.

This module is not currently in use. The Labeler class was decoupled from
the main pipeline when the label flag was removed from Lambert. It is
retained here as a reference for future work — the approach of using an
LLM to interpret and label emergent concept lattice structure remains
potentially useful, but the interface needs to be revisited alongside
the broader model architecture.
"""
from collections import defaultdict
import os, json, asyncio, aiohttp, nest_asyncio
import numpy as np

class Labeler:
    """
    Assigns human-readable labels to discovered categories using an LLM.

    Operates in three passes, each building on the last:

    1. Base labelling: labels each first-order category by its intent
       profile — the features that define it.
    2. Meta labelling: labels each unique embedding row (a group of
       co-occurring categories) as a higher-order meta-category.
    3. Iso labelling: groups meta-categories that share the same
       activation signature and labels the abstract pattern they represent.

    All LLM calls are made asynchronously via aiohttp against the
    Anthropic API. A context sentence is inferred at each pass and
    used to ground subsequent labelling prompts.

    # Not currently in use. The iso detection step was based on a
    # misunderstanding of the data — matching by activation signature
    # does not reliably identify meaningful structural equivalences.
    # The grouping itself remains useful but the interpretation needs
    # to be revisited.

    Parameters
    ----------
    categories : dict
        The category dict from CategoryExplorer.
    emb_cat : ndarray
        The final concept embedding matrix from Lambert.
    feature_labels : dict
        Maps head name to list of feature label strings.
    rep_cols : dict
        Maps head name to representative column indices.
    cat_rep_cols : list
        Representative column indices into the category matrix.
    entity_labels : list
        Human-readable names for each entity.
    eps : float, optional
        Threshold for treating a value as active. Default is 1e-3.
    """
    def __init__(self, categories, emb_cat, feature_labels, rep_cols, cat_rep_cols, entity_labels, eps=1e-3):
        self.categories     = categories
        self.emb_cat        = emb_cat
        self.feature_labels = feature_labels
        self.rep_cols       = rep_cols
        self.cat_rep_cols   = cat_rep_cols
        self.entity_labels  = entity_labels
        self.entity_type = None
        self.eps            = eps
        self.labels         = {}
        self.meta_labels    = {}
        self.isos           = {}
        self.iso_labels     = {}

    def _get_profile(self, labels, scores):
        return sorted(
            [(labels[i], round(float(scores[i]), 3))
            for i in range(len(scores)) if scores[i] > self.eps],
            key=lambda x: -x[1]
        )

    async def _infer_context(self, session, sample, prior_context=None):
        prompt = (
            f"Complete this sentence in under 20 words: 'Each category in this data represents a group of ____ sharing the same ____'. "
            f"Respond with JSON only: {{\"label\": \"...\"}}\n"
            f"Sample: {sample}\n"
        )
        if prior_context:
            prompt += f"Prior context: {prior_context}\n"
        return await self._llm_call(session, prompt)
    
    async def _gather_labels(self, session, prompts):
        results = await asyncio.gather(*[
            self._llm_call(session, prompt) for prompt in prompts.values()
        ])
        return {key: label for key, label in zip(prompts.keys(), results) if label is not None}

    def _get_intent_profiles(self):
        return {
            key: {
                name: self._get_profile(
                    [self.feature_labels[name][self.rep_cols[name][i]] for i in range(len(intent))],
                    intent
                )
                for name, (intent, _) in cat['intents'].items()
            }
            for key, cat in self.categories.items()
        }

    def _get_meta_profiles(self):
        unique_rows, inverse = np.unique(self.emb_cat, axis=0, return_inverse=True)
        cat_items = list(self.categories.items())
        cat_labels = [self.labels[cat_items[self.cat_rep_cols[i]][0]] for i in range(len(self.cat_rep_cols))]
        return {
            m: (
                self._get_profile(cat_labels, row),
                [self.entity_labels[i] for i in np.where(inverse == m)[0]]
            )
            for m, row in enumerate(unique_rows)
        }

    def _detect_isos(self):
        unique_rows, _ = np.unique(self.emb_cat, axis=0, return_inverse=True)
        groups = defaultdict(list)
        for m, row in enumerate(unique_rows):
            active = np.where(row > 0)[0]
            if len(active) == 0:
                continue
            sig = tuple(sorted(row[active].tolist()))
            groups[sig].append((m, self.meta_labels.get(m)))
        self.isos = {sig: members for sig, members in groups.items() if len(members) >= 2}

    async def _llm_call(self, session, prompt):
        for attempt in range(3):
            r = await session.post("https://api.anthropic.com/v1/messages", json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": prompt}]
            })
            data = await r.json(content_type=None)
            if "content" in data:
                return json.loads(data["content"][0]["text"].strip())["label"]
            await asyncio.sleep(2 ** attempt)
        return None
    
    async def _label_base(self, session, context):
        profiles = self._get_intent_profiles()
        prompts = {
            key: (
                f"{context}\n"
                f"Label this category in 3-6 words. Respond with JSON only: {{\"label\": \"...\"}}\n{p}"
            )
            for key, p in profiles.items()
        }
        self.labels = await self._gather_labels(session, prompts)
    
    async def _label_meta(self, session, context):
        meta_profiles = self._get_meta_profiles()
        singletons = {m: profile[0][0] for m, (profile, _) in meta_profiles.items() if len(profile) == 1}
        prompts = {
            m: (
                f"{context}\n"
                f"Label this higher-order meta-category in 3-6 words. Respond with JSON only: {{\"label\": \"...\"}}\n"
                f"Base categories: {profile}"
            )
            for m, (profile, members) in meta_profiles.items() if len(profile) > 1
        }
        self.meta_labels = {**singletons, **await self._gather_labels(session, prompts)}
    
    async def _label_isos(self, session, context):
        prompts = {
            sig: (
                f"{context}\n"
                f"What abstract pattern do these share? Give a short label (3-6 words). Respond with JSON only: {{\"label\": \"...\"}}\n"
                f"Members: {[label for _, label in members]}"
            )
            for sig, members in self.isos.items() if len(members) > 0
        }
        self.iso_labels = await self._gather_labels(session, prompts)

    async def _run(self):
        async with aiohttp.ClientSession(headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }) as session:
           
            base_context = await self._infer_context(session, {
                'entities': self.entity_labels[:5],
                'dimensions': list(self.feature_labels.keys()),
                'sample_profile': list(self._get_intent_profiles().values())[0]
            })
            print(base_context)
            await self._label_base(session, base_context)

            meta_context = await self._infer_context(session,
                {'base_labels': list(self.labels.values())[:5]}, prior_context=base_context)
            print(meta_context)
            await self._label_meta(session, meta_context)

            self._detect_isos()
            iso_context = await self._infer_context(session,
                {'meta_labels': list(self.meta_labels.values())[:5]}, prior_context=meta_context)
            print(iso_context)
            await self._label_isos(session, iso_context)

    def run(self):
        nest_asyncio.apply()
        asyncio.run(self._run())
""""
The natural language layer of the model
right now we use this to conveniently label discovered patterns and relationships
as the framework grows we can find other creative ways to weave llms into it
"""
from collections import defaultdict
import os, json, asyncio, aiohttp, nest_asyncio
import numpy as np

class Labeler:
    def __init__(self, categories, emb_cat, feature_labels, rep_cols, eps=1e-3):
        self.categories    = categories
        self.emb_cat       = emb_cat
        self.feature_labels = feature_labels
        self.rep_cols      = rep_cols
        self.eps           = eps
        self.labels        = {}
        self.meta_labels   = {}

    def _get_intent_profiles(self):
        return {
            key: {
                name: sorted([(self.feature_labels[name][self.rep_cols[name][i]], round(float(intent[i]), 3))
                              for i in range(len(intent)) if intent[i] > self.eps], key=lambda x: -x[1])
                for name, (intent, _) in cat['intents'].items()
            }
            for key, cat in self.categories.items()
        }

    def _get_meta_profiles(self):
        unique_rows, _ = np.unique(self.emb_cat, axis=0, return_inverse=True)
        return {
            m: sorted([(self.labels[key], round(float(row[i]), 3))
                       for i, (key, _) in enumerate(self.categories.items())
                       if row[i] > self.eps], key=lambda x: -x[1])
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

    async def _label_isos(self, session):
        async def _call(sig, members):
            if len(members) == 0:
                return sig, None
            member_labels = [label for _, label in members]
            label = await self._llm_call(session, 
                f"These geographic-linguistic categories share identical relational structure {sig} but over different regions.\n"
                f"Members: {member_labels}\n"
                f"What abstract pattern do they share? Give a short label (3-6 words). Respond with JSON only: {{\"label\": \"...\"}}")
            return sig, label
        results = await asyncio.gather(*[_call(sig, members) for sig, members in self.isos.items()])
        self.iso_labels = {sig: label for sig, label in results if label is not None}

    async def _llm_call(self, session, prompt):
        r = await session.post("https://api.anthropic.com/v1/messages", json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": prompt}]
        })
        data = await r.json()
        return json.loads(data["content"][0]["text"].strip())["label"]

    async def _run(self):
        profiles = self._get_intent_profiles()
        async with aiohttp.ClientSession(headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }) as session:
            # label base categories
            results = await asyncio.gather(*[
                self._llm_call(session, f"Label this geographic-linguistic category in 3-6 words. Respond with JSON only: {{\"label\": \"...\"}}\n{p}")
                for p in profiles.values()
            ])
            self.labels = dict(zip(profiles.keys(), results))

            # label meta categories
            meta_profiles = self._get_meta_profiles()
            async def _meta_call(m, profile):
                if len(profile) == 0:
                    return m, None
                if len(profile) == 1:
                    return m, profile[0][0]
                label = await self._llm_call(session, f"Label this higher-order geographic-linguistic meta-category in 3-6 words. Respond with JSON only: {{\"label\": \"...\"}}\n{profile}")
                return m, label
            meta_results = await asyncio.gather(*[_meta_call(m, p) for m, p in meta_profiles.items()])
            self.meta_labels = {m: l for m, l in meta_results if l is not None}

            # detect and label isomorphic groups
            self._detect_isos()
            await self._label_isos(session)

    def run(self):
        nest_asyncio.apply()
        asyncio.run(self._run())
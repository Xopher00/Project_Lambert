# INVARIANTS.md — Lambert Research Integration

Cross-cutting contracts for this PRD. Each invariant is machine-verifiable and enforced across all three sprints.

---

## Cite key consistency

- **Owner:** `research/bibliography.md` (defines all canonical cite keys via `<!-- [key] -->` HTML comments)
- **Preconditions:** Any research note that adds a `cite{key}` reference must use a key that exists in bibliography.md
- **Postconditions:** After Sprint 1 completes, `tools/check_citations.py` exits 0
- **Invariants:** Every `cite{key}` in any `research/` .md file has a corresponding `<!-- [key] -->` entry in `research/bibliography.md`
- **Verify:** `python tools/check_citations.py`
- **Fix:** Add the missing bibliography entry, or correct the cite key spelling

---

## Active codebase provenance-free

- **Owner:** Sprint 2 (retires the provenance package)
- **Preconditions:** Sprint 2 has completed (provenance/ moved to legacy/)
- **Postconditions:** No file outside `legacy/` imports from the provenance package
- **Invariants:** `grep -r "from provenance" . --include="*.py" | grep -v "^./legacy/" | grep -v "__pycache__"` returns nothing
- **Verify:** `bash -c 'result=$(grep -r "from provenance" . --include="*.py" | grep -v "^./legacy/" | grep -v "__pycache__"); [ -z "$result" ] && echo ok || (echo "FAIL: $result" && exit 1)'`
- **Fix:** Remove or update the import; do not add a re-export shim

---

## Provenance directory removed from project root

- **Owner:** Sprint 2
- **Preconditions:** Sprint 2 has completed
- **Postconditions:** `provenance/` directory does not exist at the project root
- **Invariants:** `test ! -d provenance`
- **Verify:** `test ! -d /home/scanbot/ua_tensors/provenance && echo ok`
- **Fix:** Ensure the directory was fully removed (not just emptied)

---

## Lambert importable

- **Owner:** `model.py` (must remain importable throughout all sprints)
- **Preconditions:** Python environment is active (`venv/`)
- **Postconditions:** `from model import Lambert` succeeds in the project root
- **Invariants:** Lambert import does not depend on `provenance/`; CategoryExplorer instantiation does not require the `_concept_fixpoint` override
- **Verify:** `cd /home/scanbot/ua_tensors && python -c "from model import Lambert; print('ok')"`
- **Fix:** Identify the broken import; do not patch by adding back provenance imports

---

## CategoryExplorer dead code absent

- **Owner:** Sprint 3
- **Preconditions:** Sprint 3 has completed
- **Postconditions:** `lattice/explorer.py` contains no `_concept_fixpoint` method and no `learn` parameter on `explore`
- **Invariants:** These two dead-code items do not exist in the file
- **Verify:** `bash -c '[ $(grep -c "_concept_fixpoint" /home/scanbot/ua_tensors/lattice/explorer.py) -eq 0 ] && [ $(grep -c "learn=" /home/scanbot/ua_tensors/lattice/explorer.py) -eq 0 ] && echo ok || echo FAIL'`
- **Fix:** Remove the method / parameter; if grep still returns hits, check for comments or docstrings referencing the name

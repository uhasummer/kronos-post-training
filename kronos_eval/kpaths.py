"""Locate sibling packages and the upstream Kronos model code without symlinks.
Search order for the upstream repo (needs model/kronos.py): $KRONOS_UPSTREAM, <root>/ (when the packages live inside the
upstream checkout), <root>/Kronos, <this package>/Kronos. <root> = $KRONOS_ROOT or this package's parent."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.environ.get("KRONOS_ROOT") or os.path.dirname(HERE)

def _is_upstream(p): return p and os.path.exists(os.path.join(p, "model", "kronos.py"))
UPSTREAM = next((p for p in (os.environ.get("KRONOS_UPSTREAM"), ROOT, os.path.join(ROOT, "Kronos"), os.path.join(HERE, "Kronos"), os.path.join(ROOT, "kronos_pretrain", "Kronos")) if _is_upstream(p)), None)
if UPSTREAM is None: raise ImportError("Kronos upstream code not found; set KRONOS_UPSTREAM or clone https://github.com/shiyu-coder/Kronos next to the packages")
for p in (UPSTREAM, os.path.join(ROOT, "kronos_pretrain"), os.path.join(ROOT, "kronos_eval"), HERE):
    if p not in sys.path: sys.path.insert(0, p)

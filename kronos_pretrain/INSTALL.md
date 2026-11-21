# Install (no symlinks needed)

```
git clone https://github.com/shiyu-coder/Kronos ~/work/Kronos          # upstream model code (or use an existing checkout)
cp -r kronos_pretrain kronos_post kronos_eval ~/work/                  # the three packages side by side
cd ~/work && pip install -r kronos_pretrain/requirements.txt
```
Every entry point does `import kpaths`, which finds the upstream code at `$KRONOS_UPSTREAM`, the packages' parent dir (when they live
inside the upstream checkout), `<parent>/Kronos`, or `<package>/Kronos`, and adds the sibling packages
to `sys.path`. Override the root with `KRONOS_ROOT=/path`. Run scripts from inside their package directory (relative data paths).
Quick checks (CPU, synthetic data): `python smoke_local.py && python smoke_opt.py && python smoke_factors.py && python smoke_merge.py`
in kronos_pretrain, then `python smoke_post.py` in kronos_post.

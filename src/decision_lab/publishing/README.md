# Publication utilities

`dataset.py` creates the normalized public data and checks both symbolic oracles.
It takes explicit source/destination paths and refuses to overwrite existing data.

`legacy_prepare.py`, `legacy_publish.py` and `legacy_archive.py` preserve older
release workflows. Their historical names, paths and cards describe those releases;
they are not the current publication entry point. Use the `v0.1.0` tag for exact
historical source. Current public assets are pinned in `release/manifest.json`.

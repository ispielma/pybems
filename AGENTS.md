# AGENTS.md

## Start here

- Read `README.md` and `pyproject.toml` before changing the package.
- Treat `src/pybeams/system.py`, `sources.py`, `elements.py`, and `plotting.py` as
  the architecture reference; read the matching files under `tests/` before
  modifying behavior.
- The local/package name is `pybeams`; the GitHub repository is currently
  `ispielma/pybems` (the spelling difference is intentional).

## Design constraints

- Model monochromatic, scalar, cylindrically symmetric fields on the radial
  order-zero QDHT grid. Keep all public lengths in SI units.
- Preserve the source/element/system object model and reuse the
  `OpticalSystem` Hankel transform. Put generally useful behavior in the package,
  without application-specific assumptions.
- For propagation changes, add deterministic numerical tests covering shapes,
  normalization or radial power, and analytic limiting cases where possible.

## Workflow

- Check `git status` first and preserve unrelated user changes.
- Run the suite from the repository root:

  ```bash
  MPLCONFIGDIR=/private/tmp/pybeams-mpl-cache \
  PYTHONPATH=src \
  /Users/ispielma/miniforge3/bin/python -m unittest discover -s tests -v
  ```

- Keep commits focused. Do not push unless the user explicitly requests it.

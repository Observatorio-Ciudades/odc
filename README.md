![PyPI](https://img.shields.io/pypi/v/your-package-name)
![Build](https://github.com/username/repo-name/actions/workflows/main.yml/badge.svg)
[![codecov](https://codecov.io/gh/username/repo-name/branch/main/graph/badge.svg)](https://codecov.io/gh/username/repo-name)
![Downloads](https://img.shields.io/pypi/dm/your-package-name)

Descripción breve de tu proyecto...


# Estructura para proyectos

_Estructura flexible para proyectos de ciencia de datos._


### Para iniciar un nuevo proyecto:
------------

    git clone This repo
    


### Estructura mínima
------------

La estructura de los folders de este proyecto es:

```
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── output            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures       <- Generated graphics and figures to be used in reporting
|   └── text          <- Reports
│
└── src                <- Source code for use in this project.
    ├── __init__.py    <- Makes src a Python module
    │
    ├── data.py        <- Scripts to download or generate data
    │
    ├── analysis.py    <- Scripts to analyse the data
    │
    └── visualization.py  <- Scripts to create exploratory and results oriented visualizations

```

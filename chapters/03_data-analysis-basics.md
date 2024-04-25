---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{code-cell}
:tags: [remove_cell]
import os
from matplotlib.pyplot import rcParams

os.chdir("..")
rcParams["figure.dpi"] = 150
```

Data Analysis in Python
=======================


## Preliminaries

```{code-cell}
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

### Using a metadata sheet

```{code-cell}
datadir = Path("data/dickinson_poetry-foundation-poems")
manifest = pd.read_csv(datadir / "manifest.csv")
```

```{code-cell}
manifest.info()
```

```{code-cell}
manifest.head()
```


### Loading a corpus

```{code-cell}
def load_corpus(datadir, files):
    """Load a corpus from file names.

    Parameters
    ----------
    datadir : Path
        Directory where the files are stored
    files : Iterable
        A list of file names

    Returns
    -------
    corpus : list
        The corpus
    """
    corpus = []
    for fname in files:
        path = datadir / fname
        with path.open("r") as fin:
            doc = fin.read()
            corpus.append(doc)

    return corpus
```

```{code-cell}
corpus = load_corpus(datadir / "poems", manifest["file"])
```

```{code-cell}
manifest["text"] = corpus.copy()
```


## Working with Tabular Data

### Series vs. DataFrame


### Indexing

**By index**

**By position**

**By condition**


### Aggregation

**Over columns**

**Over groups**


## Data Analysis

### Preprocessing

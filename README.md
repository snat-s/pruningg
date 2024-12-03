# Evolutionary Layer Pruning

## Requirements
- Python
- Poetry

## Installation
```bash
poetry install
```

## Usage

The bulk of the code is in `src/sga.py`.

To run the layer selection and subsequent fine-tuning, run:

```bash
poetry run python src/sga.py
```

We have implemented three different algorithms for pruning.
Hill Climbing, Simmulated Annealintg and Simple Genetic Algorithms.

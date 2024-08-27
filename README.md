# Evolutionary Layer Pruning

## Requirements
- Python
- Poetry

## Installation
```bash
poetry install
```

## Usage
The bulk of the code is in `src/transformer.py`. To adjust the layers of the model, change the variables:
- `num_blocks`: The number of blocks in the LLM. Eg. 8 will split a 32 layer LLM into 4 blocks of 8 layers
- `block_order`: The ordering of the `num_blocks` blocks. Eg. `[0,1,2,3,4,5,6,7]`

To run the layer selection and subsequent fine-tuning, run:
```bash
poetry run python src/transformer.py
```
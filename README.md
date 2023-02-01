## Create conda environment

```bash
conda env create --file environment.yml
```

## Update conda environment

```bash
conda env update --file environment.yml --prune
```

## Update installed cad package inside environment
```bash
conda activate cad
pip install -e .
```
# Leonardo-Da-VQGAN

[![Unix Build Status](https://img.shields.io/travis/com/Leonardo-Da-VQGAN/Leonardo-Da-VQGAN.svg?label=unix)](https://travis-ci.com/Leonardo-Da-VQGAN/Leonardo-Da-VQGAN)
[![Windows Build Status](https://img.shields.io/appveyor/ci/Leonardo-Da-VQGAN/Leonardo-Da-VQGAN.svg?label=windows)](https://ci.appveyor.com/project/Leonardo-Da-VQGAN/Leonardo-Da-VQGAN)
[![Coverage Status](https://img.shields.io/codecov/c/gh/Leonardo-Da-VQGAN/Leonardo-Da-VQGAN)](https://codecov.io/gh/Leonardo-Da-VQGAN/Leonardo-Da-VQGAN)
[![Scrutinizer Code Quality](https://img.shields.io/scrutinizer/g/Leonardo-Da-VQGAN/Leonardo-Da-VQGAN.svg)](https://scrutinizer-ci.com/g/Leonardo-Da-VQGAN/Leonardo-Da-VQGAN)
[![PyPI Version](https://img.shields.io/pypi/v/Am I Good Enough?.svg)](https://pypi.org/project/Am I Good Enough?)
[![PyPI License](https://img.shields.io/pypi/l/Am I Good Enough?.svg)](https://pypi.org/project/Am I Good Enough?)

## Environment Setup

### Environment Setup (Local development)

- Install [Conda (miniconda)](https://conda.io/miniconda.html) & [Poetry](https://python-poetry.org/docs/#installation):
- Build and activate environment:
```bash
conda env create -f environment.yml
source activate leonardo_da_vqgan
```
- Install packages:
```bash
poetry install
```
- Create an account on [Weights and Biases](https://wandb.ai)
- Setup Weights and Biases:
```bash
wandb login
```


## Commands

- Train model
```bash
poetry run train
```
- Test model
```bash
poetry run test
```
- Run pytest
```bash
poetry run pytest tests/
```

## Instructions

- To install/uninstall packages and other commands, please refer to [Poetry's documentation](https://python-poetry.org/docs/cli/)
- To run tests, please refer to [pytest's documentation](https://docs.pytest.org/en/latest/)
- To add more experiments, please refer to `config/experiments.yml`
- To run multiple experiments, please refer to `scripts/experiments.py`

## 

This project was generated with [cookiecutter](https://cookiecutter.readthedocs.io/en/1.7.2/) using [sudomaze/cookiecutter-pytorch-lightning-cluster](https://github.com/sudomaze/cookiecutter-pytorch-lightning-cluster)
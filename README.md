# Fair Synthetic Data Generation

This project studies how synthetic tabular data can be generated while preserving fairness. It compares standard generative models with fairness-aware variants on the COMPAS dataset and evaluates downstream classification performance.

## Models

- CTGAN
- TVAE
- FairGAN
- TabFairGAN

The training pipeline can be found in [Notebook_Models.ipynb](Notebook_Models.ipynb), and [fairgan.py](fairgan.py) contain our custom implementation of FairGAN based on the model presented in [(Xu et al. 2018)](#Bibliography).

## Dataset

The project uses the COMPAS recidivism dataset from [data/compas-scores-two-years.csv](data/compas-scores-two-years.csv). The main sensitive attribute used in the study is `sex`, and the target variable is `is_recid`.

A breakdown of the attributes used in the study can be found [HERE](data/COMPAS.md)

## Outputs

The repository generates synthetic data for multiple models and fairness settings, including files such as:

- `ctgan_synthetic.csv`
- `tvae_synthetic.csv`
- `FairGAN_lambda_0.0_synthetic.csv`
- `TabFairGAN_lambda_1.0_synthetic.csv`

Evaluation plots and summaries are stored in [figures/](figures/).

## Notes

This project is focused on fairness-aware synthetic data generation for tabular data, with emphasis on comparing utility and bias trade-offs across different generators.

## Bibliography
Xu, D., Yuan, S., Zhang, L., & Wu, X. (2018, December). Fairgan: Fairness-aware generative adversarial networks. In 2018 IEEE international conference on big data (big data) (pp. 570-575). IEEE.

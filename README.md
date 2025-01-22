# Auto-ADMET
An Effective and Interpretable Evolutionary-based AutoML Method for Chemical ADMET Property Prediction


## Description


This repository contains code and data for Auto-ADMET's work, which is an Automated Machine Learning (AutoML) method targetting personalised predictions for Absorption, Distribution, Metabolism, Excretion, and Toxicity (ADMET) properties.

For this work, an evolutionary-based  method (i.e., grammar-based genetic programming) is used to search and optimise machine learning (ML) pipelines in the context of chemical ADMET property prediction. Chemical Representation, Feature Scaling, Feature Selection, and ML Modelling are considered to compose the predictive ML-driven pipelines for ADMET.

## How to Install?

Our method uses Anaconda to install the requirements. It is worth noting we are relying on [alogos](https://github.com/robert-haas/alogos) for the basics of GGP.

`conda env create -f requirements.yaml`


## How to use the installed conda environment?

`conda activate automl4pk`

## How to use the AutoML method considering the Python code available?

After activating automl4pk environment, run:

`python automl4pk.py training_file.csv testing_file.csv seed_number num_cores output_dir`

E.g., using:

* "datasets/01_caco2_train.csv" as the training file.csv
* "datasets/01_caco2_blindtest.csv" as the testing file.csv
* "." as the output directory (output_dir)

`python automl4pk.py datasets/01_caco2_train.csv datasets/01_caco2_blindtest.csv .`

Optional parameters can also be used:

* population size (pop_size). Default value: 30.
* crossover rate (xover_rate). Default value: 0.9.
* mutation rate (mut_rate). Default value: 0.1.
* time to run the AutoML method (time_budget_min). Default value: 60 (min).
* time to run each algorithm/pipeline (time_budget_minutes_alg_eval). Default value: 5 (min).
* Random seed (seed). Default value: 42.
* Number of cores (num_cores). Default value: 1.


`python automl4pk.py datasets/01_caco2_train.csv datasets/01_caco2_blindtest.csv . -pop_size 30 -xover_rate 0.9 -mut_rate 0.1 -time_budget_min 60 -time_budget_minutes_alg_eval 5 -seed 42 -num_cores 1`


## Publication
Alex de Sá, and David Ascher, Towards Evolutionary-based Automated Machine Learning for Small Molecule Pharmacokinetic Prediction, GECCO Companion (ECADA workshop), 2024.​ https://dl.acm.org/doi/abs/10.1145/3638530.3664166

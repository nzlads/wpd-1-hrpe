# Western Power Distribution Data Challenge 1: High Resolution Peak Estimation

This is the data for the first of three Western Power Distribution (WPD) short data challenges! The aims of these
challenges include:

- Demonstrating the value in making data openly available
- Increasing the visibility of some of the challenges network operators face
- Increase the understanding of the different ways to tackle some of these problems
- Providing high quality and accurate benchmarks with which to enable innovation and research

Note all the slides from the kick-off and other information can be found on our LinkedIn
group: https://www.linkedin.com/groups/9025332/

This initial challenge aims to understand how accurately high resolution features can be estimated given only
information from lower resolution data. Specifically we are asking participants to estimate the highest peak value and
lowest trough at a one minute resolution within each half hourly period given only half hourly measurements. This is an
interesting problem to a distribution network operator as the spikes in demand can mean strain on their network. Such
issues may become increasingly common, especially on the lower voltages of the network, due to the expanding use of
lower carbon technologies such as electric vehicles, and heat pumps. However, monitoring can be expensive (especially in
the long term) as it requires investment in additional storage, communications equipment and processing units.

Project Organization
------------

    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── explorations       <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                      the creator's initials, and a short `-` delimited description, e.g.
    │                      `1.0-jqp-initial-data-exploration`.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

--------



## Setup


```python
python -m pip install poetry
# or 
pipx install poetry

## Verify
poetry --version

```


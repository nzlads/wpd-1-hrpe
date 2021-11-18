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

## Setup

```sh
# Poetry's preferred install method on windows via powershell
(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -
# Otherwise manual install with
python -m pip install poetry
# or 
pipx install poetry

## Verify
poetry --version

# virtual env
python -m venv venv
venv\Scripts\activate

poetry install
```

### Poetry Commands

https://hackersandslackers.com/python-poetry-package-manager/


### Weather data

https://codalab.lisn.upsaclay.fr/competitions/213#participate-get_starting_kit

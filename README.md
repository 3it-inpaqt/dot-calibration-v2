# Quantum dot auto tuning V2

Quantum dot autotuning with machine learning V2. First version
available [here](https://github.com/3it-nano/dot-calibration).

Using [QDSD](https://github.com/3it-nano/qdsd-dataset) dataset.

## Install

Required `python >= 3.8` and `pip`

```shell script
pip install -r requirements.txt
```

Then download the data set
from [this private Teams folder](https://usherbrooke.sharepoint.com/:f:/r/sites/UdeS-UW-Memristor-basedMLforQuantumTechs/Documents%20partages/General/Datasets/QDSD?csf=1&web=1&e=YtBFnn)
and unzip in into a `data` folder at the root of this project. The mandatory files are:

* interpolated_csv.zip
* labels.json

## Settings

Create a file `settings.yaml` to override settings documented in `utils/settings.py`

**For example**:

```yaml
run_name: tmp
seed: 0
logger_console_level: info
show_images: False
model_type: CNN
trained_network_cache_path: out/cnn/best_network.pt
normalization_values_path: out/cnn/normalization.yaml
nb_epoch: 300
```

## Start run

### Line classification task

```shell
python3 start_lines.py
```

### Charge state autotuning task

```shell
python3 start_tuning.py
```

## Files structure

**Code**:

* `autotuning/` : The different autotuning algorithm implementations
* `baselines/` : Line detection baselines (not machine learning models)
* `classes/` : Custom classes definition
* `datasets/` : Dataset in pyTorch format (only [QDSD](https://github.com/3it-nano/qdsd-dataset) so far)
* `networks/` : Neural networks definition in pyTorch format
* `plots/` : Code to generate figures
* `runs/` : Code logic for the execution of the different tasks
* `utils/` : Utility code for text output
* `start_lines.py` : Main file to start the line classification task
* `start_tuning.py` : Main file to start the charge state autotuning task
* `settings.yaml` : Projet configuration file (**should be created by the user**)

**Others**:

* `data/` : Contains diagrams data (**should be downloaded by the user**) and generated cache files
* `out/` : Generated directory that contains run results log and plots if `run_name` setting field is defined

## Credits

* [PyTorch](https://pytorch.org/)
* [Blitz - Bayesian Layers in Torch Zoo](https://github.com/piEsposito/blitz-bayesian-deep-learning)
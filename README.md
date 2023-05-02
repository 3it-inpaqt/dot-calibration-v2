# Quantum dot auto tuning V2

Quantum dot autotuning with machine learning V2. First version
available [here](https://github.com/3it-inpaqt/dot-calibration).

Using [QDSD](https://github.com/3it-inpaqt/qdsd-dataset) dataset.

## Install

Required `python >= 3.8` and `pip`

```shell script
pip install -r requirements.txt
```

Then download the data set
from [this private Teams folder](https://usherbrooke.sharepoint.com/:f:/r/sites/GroupeNano/Documents%20partages/Dataset%20Machine%20Learning/QDSD?csf=1&web=1&e=NyI7i5)
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
python3 start_tuning_offline.py
```

## Files structure

### Code

* `autotuning/` : The different autotuning algorithm implementations
* `classes/` : Custom classes and data structure definition
* `connectors/` : Interface to connect with experimental measurement tools (for online diagrams tuning)
* `datasets/` : Diagrams loading (from [QDSD](https://github.com/3it-inpaqt/qdsd-dataset)) and datasets in pyTorch
  format
* `models/` : Neural network definitions in pyTorch format and baseline models
* `documentation/` : Documentation and process description
* `plots/` : Code to generate figures
* `runs/` : Code logic for the execution of the different tasks
* `utils/` : Miscellaneous utility code (output handling, settings, etc.)
* `start_full_exp.py`: Script to run the complete experiment benchmark (line and autotuning tasks repeated with
  different meta-parameters)
* `start_tasks_planner.py`: Script to automatise several benchmarks with grid-search
* `start_lines.py` : Main file to start the line classification task
* `start_tuning_[online|offline].py` : Main files to start the charge state autotuning task (either online or offline)

### Created files

* `data/` : Contains diagrams data (**should be downloaded by the user**) and generated cache files
* `out/` : Generated directory that contains run results log and plots if `run_name` setting field is defined
* `settings.yaml` : Projet configuration file (**should be created by the user**)

## Workflow

![code-workflow](documentation/workflow.drawio.svg "High-level representation of the code Workflow")
_Note: The not-solid lines represent work in progress._

## Credits

* [PyTorch](https://pytorch.org/)
* [Blitz - Bayesian Layers in Torch Zoo](https://github.com/piEsposito/blitz-bayesian-deep-learning)
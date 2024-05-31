# Quantum dot autotuning V2

Quantum dot autotuning with machine learning V2. First version
available [here](https://github.com/3it-inpaqt/dot-calibration) (private repository).

The model training and the offline autotuning simulations use the [QDSD](https://doi.org/10.5281/zenodo.11402792)
dataset, which is generated using [this repository](https://github.com/3it-inpaqt/qdsd-dataset).

## Install

Required `python >= 3.10` and `pip`

```shell script
pip install -r requirements.txt
```

Then download the [QDSD](https://doi.org/10.5281/zenodo.11402792) dataset
and unzip in into a `data` folder at the root of this project. The mandatory files are:

* data/interpolated_csv.zip
* data/labels.json

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
dropout: 0.6
nb_train_update: 30000
```

## Start run

### Line classification task (NN train & test)

```shell
python3 start_lines.py
```

> **Note**: The [QDSD](https://doi.org/10.5281/zenodo.11402792) dataset should be downloaded and extracted in the `data`
> folder.

### Offline charge autotuning

```shell
python3 start_tuning_offline.py
```

> **Note**: If the `trained_network_cache_path` setting is not set, the script will run the line classification task
> first to train a new model.

### Online charge autotuning

```shell
python3 start_tuning_online.py
```

> **Note**: The `trained_network_cache_path` need to be set.
>
> A `connectors` need to be implemented to communicate with the experimental equipment.
> See [connectors/py_hegel.py](connectors/py_hegel.py) for an example.

### Reproduce the results of the paper

```shell
python3 start_full_exp.py --seed 42000
```

> **Note**: The `--seed` argument could be incremented to repeat the experiment with different random seeds.
>
> Running this script can take several days (3 days with a GPU 3070Ti).

## Files structure

### Repository files

* `autotuning/` : The different autotuning algorithm implementations
* `circuit_simulation/` : Code to generate circuit description, run circuit simulation and benchmark the results
* `classes/` : Custom classes and data structure definition
* `connectors/` : Interface to connect with experimental measurement tools (for online diagrams tuning)
* `datasets/` : Diagrams loading and datasets in pyTorch
  format
* `models/` : Neural network definitions in pyTorch format and baseline models
* `documentation/` : Documentation and process description
* `plots/` : Code to generate figures
* `runs/` : Code logic for the execution of the different tasks
* `utils/` : Miscellaneous utility code (output handling, settings, etc.)
* `start_full_exp.py`: Script to run the complete experiment benchmark (line and autotuning tasks repeated with
  different meta-parameters)
* `start_tasks_planner.py`: Script to automatize several benchmarks with grid-search
* `start_lines.py` : Main file to start the line classification task
* `start_tuning_[online|offline].py` : Main files to start the charge state autotuning task (either online or offline)

### Created files

* `data/` : Contains [QDSD](https://doi.org/10.5281/zenodo.11402792) diagrams data (**should be downloaded by the user
  **) and generated cache files.
* `out/` : Generated directory that contains run results log and plots if `run_name` setting field is defined. The
  outputs from the paper can be downloaded [here](https://doi.org/10.5281/zenodo.11403192).
* `settings.yaml` : Projet configuration file (**should be created by the user**)

## Workflow

![code-workflow](documentation/workflow.drawio.svg "High-level representation of the code Workflow")

## Credits

* [PyTorch](https://pytorch.org/)
* [Blitz - Bayesian Layers in Torch Zoo](https://github.com/piEsposito/blitz-bayesian-deep-learning)

# Contributors

* [Victor Yon](https://github.com/victor-yon): Main developer
* [Bastien Galaup](https://github.com/assh2802): Contributor
* [Yohan Finet](https://github.com/YohanFinet): Integrate circuit simulation and hardware-aware methods
# Quantum dot auto tuning V2

Quantum dot auto tuning with machine learning V2.

First version available [here](https://github.com/3it-nano/dot-calibration).

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

## Credits

* [PyTorch](https://pytorch.org/)
* [Blitz - Bayesian Layers in Torch Zoo](https://github.com/piEsposito/blitz-bayesian-deep-learning)
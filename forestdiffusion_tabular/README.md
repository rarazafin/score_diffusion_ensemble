# ForestDiffusion on tabular data

The code is based on [this repository](https://github.com/SamsungSAILMontreal/ForestDiffusion).

To install the necessary Python packages, run:
```sh
pip install -r requirements.txt
```

To train and test our methods, make sure to navigate to the subdirectory, download the dataset, store it in a `dataset` folder, and run the following script
```sh
python script_genetation.py --methods forest_diffusion --diffusion_type vp --forest_model random_forest --agg arithmetic --dataset airfoil_self_noise --n_batch 0 --nexp 3 --n_estimators 100 --ycond=False
```
where `--n_estimators` is the desired number of trees and `--agg` is the aggregation scheme.

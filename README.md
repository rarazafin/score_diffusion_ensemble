# When are Two Scores Better than One? Investigating Ensembles of Diffusion Models

This is an initial version of the repository and will be completed for the final release.  

## Repository structure

The repository is organized into three main folders, each corresponding to a specific experimental setup:

- **`score_based_cifar10/`**  
  Contains experiments on the **CIFAR-10** dataset using **Score-Based Diffusion Models (SDE formulation [1])**. 

- **`ddim_ffhq/`**  
  Contains experiments on the **FFHQ 256×256** dataset using **Deterministic Diffusion Implicit Models (DDIM [2])**.

- **`forestdiffusion_tabular/`**  
  Contains experiments related to **ForestDiffusion** on **tabular data [3]**.

The implementations closely follow the authors’ original implementations. 

[1] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole.
Score-based generative modeling through stochastic differential equations. In International Conference on
Learning Representations, 2021b.

[2] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. International Con-
ference on Learning Representations, 2020.

[3] Alexia Jolicoeur-Martineau, Kilian Fatras, and Tal Kachman. Generating and imputing tabular data via
diffusion and flow-based gradient-boosted trees. In International Conference on Artificial Intelligence and
Statistics, pp. 1288–1296. PMLR, 2024.

## How to run the code

### Dependencies

Each subdirectory contains its own dependencies and a dedicated requirements.txt file. To install the necessary Python packages for a given experiment, navigate into the corresponding folder and run:
```sh
pip install -r requirements.txt
```


### Score-based diffusion models on CIFAR-10

The code is based on [this repository](https://github.com/yang-song/score_sde_pytorch).

We provide [checkpoints](https://drive.google.com/drive/folders/1uprGLUfGtMx__woTAnPtlj0TGYP2ZIpQ?usp=drive_link) corresponding to models trained for 200k iterations in this setting.
Reference statistics required for quantitative evaluation (along with the corresponding computation code) can be found in the original repository. Once the [statistics file](https://drive.google.com/drive/folders/1uprGLUfGtMx__woTAnPtlj0TGYP2ZIpQ?usp=drive_link) have been downloaded, place them in the `assets/stats` directory.

Below are example flag configurations used for our experiments to measure perceptual quality on deep ensemble. Use `cd` to navigate to the appropriate subdirectory before following the run instruction below.

Indicate the name of the checkpoint file (without the .pth extension) with the flag `--ckpt`. The aggregation scheme is set via `--agg`.

* Base flags
```sh
BASE_FLAGS="--config=configs/vp/cifar10_ddpmpp_continuous.py --workdir='' --agg=arithmetic --num_samples=10000"
```

* No ensemble
```sh
XP_FLAGS="--n_ensembles=0 --ensemble_type=none
```
* Ensemble $K=2$
```sh
XP_FLAGS="--n_ensembles=2 --ensemble_type=deep_ensemble --config.seed=0"
```
The seed controls which subset of models (among the 5 available) is selected.

* Ensemble $K=5$
```sh
XP_FLAGS="--n_ensembles=5 --ensemble_type=deep_ensemble"
```

Then run:
```sh
python ensemble_FID_IS_KID.py $BASE_FLAGS $XP_FLAGS
```

### DDIM on FFHQ

The code is based on [this repository](https://github.com/openai/guided-diffusion).

We provide similarly to CIFAR-10 the [checkpoints](https://drive.google.com/drive/folders/1uprGLUfGtMx__woTAnPtlj0TGYP2ZIpQ?usp=drive_link) of the 4 models. Stats file is available [here](https://drive.google.com/drive/folders/1uprGLUfGtMx__woTAnPtlj0TGYP2ZIpQ?usp=drive_link) and computed similarly to CIFAR-10's ones.

Below are example flag configurations used for our experiments to measure perceptual quality on deep ensemble. Use `cd` to navigate to the appropriate subdirectory before following the run instruction below.

* Base flags
```sh
BASE_FLAGS="--config='configs/ffhq.yaml' --workdir='' --batch_size=32 --eta=0.5 --num_sampling_timesteps=10 --num_samples=10000"
```
where `--num_sampling_timesteps` is the number of DDIM steps and `--eta` is the level of entropy.

* No ensemble
```sh
XP_FLAGS="--n_ensembles=0 --ensemble_type='none'"
```
* Ensemble $K=2$
```sh
XP_FLAGS="--n_ensembles=2 --ensemble_type='deep_ensemble' --agg='arithmetic' --seed=0"
```
The seed controls which subset of models (among the 5 available) is selected.

* Ensemble $K=5$
```sh
XP_FLAGS="--n_ensembles=5 --ensemble_type='deep_ensemble' --agg='arithmetic'"
```

Then run:
```sh
python test.py $BASE_FLAGS $XP_FLAGS
```

### ForestDiffusion on tabular data

The code is based on [this repository](https://github.com/SamsungSAILMontreal/ForestDiffusion).

To train and test our methods, make sure to navigate to the subdirectory, download the dataset, store it in a `dataset` folder, and run the following script
```sh
python script_genetation.py --methods forest_diffusion --diffusion_type vp --forest_model random_forest --agg arithmetic --dataset airfoil_self_noise --n_batch 0 --nexp 3 --n_estimators 100 --ycond=False
```
where `--n_estimators` is the desired number of trees and `--agg` is the aggregation scheme.



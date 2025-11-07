# DDIM on FFHQ

The code is based on [this repository](https://github.com/openai/guided-diffusion).

We provide similarly to CIFAR-10 the [checkpoints](https://drive.google.com/drive/folders/1uprGLUfGtMx__woTAnPtlj0TGYP2ZIpQ?usp=drive_link) of the 4 models. Stats file is available [here](https://drive.google.com/drive/folders/1uprGLUfGtMx__woTAnPtlj0TGYP2ZIpQ?usp=drive_link) and computed similarly to CIFAR-10's ones.

To install the necessary Python packages, run:
```sh
pip install -r requirements.txt
```

Below are example flag configurations used for our experiments to measure perceptual quality on deep ensemble.

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

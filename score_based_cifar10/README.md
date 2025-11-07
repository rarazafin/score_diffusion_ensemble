# Score-based diffusion models on CIFAR-10

The code is based on [this repository](https://github.com/yang-song/score_sde_pytorch).

We provide [checkpoints](https://drive.google.com/drive/folders/1uprGLUfGtMx__woTAnPtlj0TGYP2ZIpQ?usp=drive_link) corresponding to models trained for 200k iterations in this setting.
Reference statistics required for quantitative evaluation (along with the corresponding computation code) can be found in the original repository. Once the [statistic files](https://drive.google.com/drive/folders/1uprGLUfGtMx__woTAnPtlj0TGYP2ZIpQ?usp=drive_link) have been downloaded, place them in the `assets/stats` directory.

To install the necessary Python packages, run:
```sh
pip install -r requirements.txt
```

Below are example flag configurations used for our experiments to measure perceptual quality on deep ensemble. Use `cd` to navigate to the appropriate subdirectory before following the run instruction below.

Indicate the name of the checkpoint file (without the .pth extension) with the flag `--ckpt`. The aggregation scheme is set vi `--agg`.

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

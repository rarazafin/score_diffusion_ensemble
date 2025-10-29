## 🗂 Repository structure

This is an initial version of the repository and will be completed for the final release.  It is organized into three main folders, each corresponding to a specific experimental setup:

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


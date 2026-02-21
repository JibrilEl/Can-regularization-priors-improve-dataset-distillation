# Can regularization priors improve dataset distillation ?

The goal of this repo is to reproduce dataset distillation [1] and enhance the method with different optimization priors.

![Distillation without prior](assets/without_prior.png?raw=true)

*Distillation without prior*

![Distillation with sparsity prior](assets/with_prior.png?raw=true)

*Distillation with sparsity prior*

## Report

Full report with ablation studies is available [![here](assets/poster_preview.png)](assets/report.pdf).

## Distilling data

Example runs to distill MNIST :

```
# Basic run (no prior, 10 images)
python dataset_distillation.py --prior none --n_image 10 --n_iter 600

# Smoothness prior, 200 images
python dataset_distillation.py --prior smoothness --n_image 200 --n_iter 2000 --n_models 5 --lambda_param 0.002

# Sparsity prior
python dataset_distillation.py --prior sparsity --n_image 10 --n_iter 2000 --lambda_param 0.00001

# Distillation prior (requires pretrained model)
python dataset_distillation.py --prior distill --n_image 10 --n_iter 1000 --lambda_param 0.02 --pretrained_path ./pretrained_model_state_dict.pt

```

## Jupyter notebooks

The python script was built from [![this notebook]()](dataset_distillation.ipynb). It contains numerous plots if you do not want to rerun the experiments yourself.


## References

1. WANG, Tongzhou, ZHU, Jun-Yan, TORRALBA, Antonio, et al. Dataset distillation. arXiv preprint arXiv:1811.10959, 2018.
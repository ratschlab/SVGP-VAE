# Scalable Gaussian Process VAE

Code for paper [Scalable Gaussian Process Variational Autoencoders](https://arxiv.org/abs/2010.13472). 

Initially forked from [this cool repo](https://github.com/scrambledpie/GPVAE).

## Dependencies
* Python >= 3.6
* TensorFlow = 1.15
* TensorFlow Probability = 0.8

## Setup 
1. Clone or download this repo. `cd` yourself to it's root directory.
2. Grab or build a working python enviromnent. [Anaconda](https://www.anaconda.com/) works fine.
3. Install dependencies, using `pip install -r requirements.txt`
4. Test the setup by running `python BALL_experiment.py --elbo VAE`
## Experiments
Here we report run configurations which were used to produce results presented in the paper. 
For all available configurations run
`python --BALL_experiment.py --help`
or
`python --MNIST_experiment.py --help`
or 
`python --SPRITES_experiment.py --help`.
### Moving ball

##### VAE 
`python BALL_experiment.py --elbo VAE`

##### [GPVAE_Pearce](http://proceedings.mlr.press/v118/pearce20a/pearce20a.pdf)
`python BALL_experiment.py --elbo GPVAE_Pearce`

##### SVGPVAE
`python BALL_experiment.py --elbo SVGPVAE_Hensman --clip_qs`

### Rotated MNIST

##### CVAE
`python MNIST_experiment.py --elbo CVAE `

##### [GPVAE_Casale](https://arxiv.org/abs/1810.11738)
`python MNIST_experiment.py --elbo GPVAE_Casale --GP_joint --ov_joint --clip_qs --opt_regime VAE-100 GP-100 --PCA`

#### [SVIGP](https://arxiv.org/abs/1309.6835)
`python MNIST_experiment.py --elbo SVIGP_Hensman --ip_joint --GP_joint --ov_joint --clip_qs --PCA --nr_epochs 2000`

##### SVGPVAE
`python MNIST_experiment.py --elbo SVGPVAE_Hensman --ip_joint --GP_joint --ov_joint --clip_qs --GECO --PCA`

To generate other rotated MNIST datasets use `generate_rotated_MNIST` function in `utils.py`.

### SPRITES dataset
To generate SPRITES dataset:
- clone the original [SPRITES repo](https://github.com/YingzhenLi/Sprites)
- set the SPRITES repo path on line 5 in SPRITES_utils.py
- run `python SPRITES_utils.py`

To run SPRITES experiment:

`python SPRITES_experiment.py --elbo SVGPVAE_Hensman --ip_joint --GPLVM_joint --PCA --clip_qs --GECO --object_kernel_normalize --clip_grad`

## Authors
- Metod Jazbec (jazbec.metod@gmail.com)

## Misc
If you want to see yet another cool GP-VAE model, check out [this](https://github.com/metodj/FGP-VAE).

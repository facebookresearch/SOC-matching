# Stochastic Optimal Control Matching

This repo contains the code for the paper [Stochastic Optimal Control Matching](https://arxiv.org/pdf/2312.02027.pdf). We propose the SOCM training loss to learn controls in stochastic optimal control problems. We compare it to the following existing losses: the relative entropy loss, the cross-entropy loss, the log-variance loss, the moment loss and the variance loss. We also compare it to a version of the SOCM loss where the matrices $M$ are fixed to the identity and not learned. We used Python 3.9.15, and the following versions of libraries:
* torch: 1.13.1
* hydra: 1.3.1
* omegaconf: 2.3.0
* matplotlib: 3.6.2
* numpy: 1.23.5

The commands to run all the algorithms and to obtain the plots can be found below.

## Quadratic OU easy
To run the algorithms:

`python main.py method.algorithm='SOCM','SOCM_const_M','rel_entropy','cross_entropy','log-variance','moment','variance' method.lmbd=1.0 method.setting='OU_quadratic_easy' method.gamma=2.0 method.scaling_factor_M=0.1 optim.M_lr=1e-3 optim.batch_size=128 method.num_iterations=60000 -m`

To get the plots:

`python plots.py method.lmbd=1.0 method.setting='OU_quadratic_easy' method.gamma=2.0 method.scaling_factor_M=0.1 optim.M_lr=1e-3 optim.batch_size=128`

## Quadratic OU hard, warm start
To run the algorithms:

`python main.py method.algorithm='SOCM','SOCM_const_M','rel_entropy','cross_entropy','log-variance','moment','variance' method.lmbd=1.0 method.setting='OU_quadratic_hard' method.gamma=2.0 method.scaling_factor_M=0.1 method.scaling_factor_nabla_V=0.1 optim.M_lr=1e-2 method.use_warm_start=True method.num_iterations_splines=60000 optim.splines_lr=3e-4 method.num_steps=150 optim.batch_size=64 method.num_iterations=60000 arch.hdims_M=[128,128] -m`

To get the plots:

`python plots.py method.lmbd=1.0 method.setting='OU_quadratic_hard' method.gamma=2.0 method.scaling_factor_M=0.1 method.scaling_factor_nabla_V=0.1 optim.M_lr=1e-2 method.use_warm_start=True method.num_iterations_splines=60000 optim.splines_lr=3e-4 method.num_steps=150 optim.batch_size=64 method.num_iterations=60000`

## Quadratic OU hard, no warm start
To run the algorithms:

`python main.py method.algorithm='SOCM','SOCM_const_M','rel_entropy','cross_entropy','log-variance','moment','variance' method.lmbd=1.0 method.setting='OU_quadratic_hard' method.gamma=2.0 method.scaling_factor_M=0.1 optim.M_lr=1e-3 optim.batch_size=128 method.num_iterations=80000 -m`

To get the plots:

`python plots.py method.lmbd=1.0 method.setting='OU_quadratic_hard' method.gamma=2.0 method.scaling_factor_M=0.1 optim.M_lr=1e-3 optim.batch_size=128 method.num_iterations=80000`

## Linear OU
To run the algorithms:

`python main.py method.algorithm='SOCM','SOCM_const_M','rel_entropy','cross_entropy','log-variance','moment','variance' method.d=10 method.lmbd=1.0 method.gamma=2.0 method.setting='OU_linear' method.num_steps=100 method.scaling_factor_M=0.1 optim.M_lr=1e-3 optim.batch_size=64 method.num_iterations=60000 -m`

To get the plots:

`python plots.py method.d=10 method.lmbd=1.0 method.gamma=2.0 method.setting='OU_linear' method.num_steps=100 method.scaling_factor_M=0.1 optim.M_lr=1e-3 optim.batch_size=64 method.num_iterations=60000`

## Double Well
To run the algorithms:

`python main.py method.algorithm='SOCM','SOCM_const_M','rel_entropy','cross_entropy','log-variance','moment','variance' method.lmbd=1.0 method.gamma=6.0 method.setting='double_well' method.d=10 method.num_steps=200 method.delta_t_optimal=0.001 method.delta_x_optimal=0.001 method.n_samples_control=65536 method.scaling_factor_M=0.1 optim.M_lr=1e-3 optim.batch_size=128 method.num_iterations=60000 method.seed=0 -m`

To get the plots:

`python plots.py method.lmbd=1.0 method.gamma=6.0 method.setting='double_well' method.d=10 method.num_steps=200 method.delta_t_optimal=0.001 method.delta_x_optimal=0.001 method.n_samples_control=65536 method.scaling_factor_M=0.1 optim.M_lr=1e-3 optim.batch_size=128 method.num_iterations=60000 method.seed=0`

## Molecular dynamics
To run the algorithms:
`python main.py method.algorithm='SOCM' method.lmbd=1.0 method.setting='molecular_dynamics' method.d=1 method.gamma=2.0 method.gamma2=2.0 method.scaling_factor_M=0.1 optim.M_lr=1e-3 optim.batch_size=128 method.num_iterations=60000 method.use_stopping_time=True optim.batch_size=64 arch.hdims_M=[64,64] -m`
`python main.py method.algorithm='SOCM','rel_entropy','cross_entropy','log-variance','moment','variance' method.lmbd=1.0 method.setting='molecular_dynamics' method.d=1 method.gamma=2.0 method.gamma2=2.0 method.scaling_factor_M=0.1 optim.M_lr=1e-3 method.num_iterations=60000 method.use_stopping_time=True method.num_steps=50 optim.batch_size=64 arch.hdims_M=[64,64] -m`


## Citations
If you find this repository helpful for your publications,
please consider citing our paper:

```
@misc{domingoenrich2023stochastic,
      title={Stochastic Optimal Control Matching}, 
      author={Carles Domingo-Enrich and Jiequn Han and Brandon Amos and Joan Bruna and Ricky T. Q. Chen},
      year={2023},
      eprint={2312.02027},
      archivePrefix={arXiv},
      primaryClass={math.OC}
}
```

## License
This repository is licensed under the
[CC BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/).

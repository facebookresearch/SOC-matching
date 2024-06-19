## Quadratic OU easy
To run the algorithms:

<!-- `python main.py method.algorithm='SOCM','UW_SOCM','UW_SOCM_diag','UW_SOCM_diag_2B','SOCM_const_M','SOCM_adjoint','work_adjoint','discrete_adjoint','cross_entropy','log-variance','moment','variance','reinf','reinf_fr','SOCM_cost','SOCM_cost_diag','SOCM_cost_diag_2B','reinf_unadj','SOCM_work','SOCM_work_diag','SOCM_work_diag_2B' method.lmbd=1.0 method.setting='OU_quadratic_easy' method.gamma=2.0 method.scaling_factor_M=0.1 optim.M_lr=1e-3 optim.batch_size=128 method.num_iterations=60000 method.save_every=10 -m` -->
`python main.py method.algorithm='SOCM','UW_SOCM','SOCM_identity','UW_SOCM_identity','SOCM_diag','UW_SOCM_diag','SOCM_diag_2B','UW_SOCM_diag_2B','SOCM_sc','UW_SOCM_sc','SOCM_sc_2B','UW_SOCM_sc_2B','SOCM_adjoint','work_adjoint','work_adjoint_STL','continuous_adjoint','continuous_adjoint_STL','discrete_adjoint','discrete_adjoint_STL','cross_entropy','log-variance','moment','variance','reinf','reinf_fr','SOCM_cost','SOCM_cost_STL','SOCM_cost_diag','SOCM_cost_diag_STL','SOCM_cost_diag_2B','SOCM_cost_diag_2B_STL','reinf_unadj','SOCM_work','SOCM_work_STL','SOCM_work_diag','SOCM_work_diag_STL','SOCM_work_diag_2B','SOCM_work_diag_2B_STL' method.lmbd=1.0 method.setting='OU_quadratic_easy' method.gamma=2.0 method.scaling_factor_M=0.1 optim.M_lr=3e-4 optim.batch_size=128 method.num_iterations=60000 method.save_every=10 method.efficient_memory=True method.output_matrix=False hydra.sweep.dir='./outputs_2/multiruns' hydra.sweep.subdir='OU_quadratic_easy' -m`

`python main.py method.algorithm='SOCM_cost_diag_STL' method.lmbd=1.0 method.setting='OU_quadratic_easy' method.gamma=2.0 method.scaling_factor_M=0.1 optim.M_lr=3e-4 optim.batch_size=128 method.num_iterations=60000 method.save_every=10 method.efficient_memory=True method.output_matrix=False method.n_samples_control=8192`

To get the plots:

`python plots.py method.lmbd=1.0 method.setting='OU_quadratic_easy' method.gamma=2.0 method.scaling_factor_M=0.1 optim.M_lr=3e-4 optim.batch_size=128 method.plot_number=1`

## Quadratic OU hard, no warm start
To run the algorithms:

`python main.py method.algorithm='SOCM','UW_SOCM','UW_SOCM_diag','UW_SOCM_diag_2B','SOCM_const_M','SOCM_adjoint','work_adjoint','discrete_adjoint','cross_entropy','log-variance','moment','variance','reinf','reinf_fr','SOCM_cost','SOCM_cost_diag','SOCM_cost_diag_2B','reinf_unadj','SOCM_work','SOCM_work_diag','SOCM_work_diag_2B' method.lmbd=1.0 method.setting='OU_quadratic_hard' method.gamma=2.0 method.scaling_factor_M=0.1 optim.M_lr=3e-4 optim.batch_size=128 method.num_iterations=80000 method.save_every=10 -m`

To get the plots:

`python plots.py method.lmbd=1.0 method.setting='OU_quadratic_hard' method.gamma=2.0 method.scaling_factor_M=0.1 optim.M_lr=3e-4 optim.batch_size=128 method.num_iterations=80000 method.plot_number=1`

## Linear OU
To run the algorithms:

`python main.py method.algorithm='SOCM','UW_SOCM','UW_SOCM_diag','UW_SOCM_diag_2B','SOCM_const_M','SOCM_adjoint','work_adjoint','discrete_adjoint','cross_entropy','log-variance','moment','variance','reinf','reinf_fr','SOCM_cost','SOCM_cost_diag','SOCM_cost_diag_2B','reinf_unadj','SOCM_work','SOCM_work_diag','SOCM_work_diag_2B' method.d=10 method.lmbd=1.0 method.gamma=2.0 method.setting='OU_linear' method.num_steps=100 method.scaling_factor_M=0.1 optim.M_lr=3e-4 optim.batch_size=64 method.num_iterations=60000 method.save_every=10 -m`

To get the plots:

`python plots.py method.d=10 method.lmbd=1.0 method.gamma=2.0 method.setting='OU_linear' method.num_steps=100 method.scaling_factor_M=0.1 optim.M_lr=3e-4 optim.batch_size=64 method.num_iterations=60000 method.plot_number=1`

## Double Well
To run the algorithms:

`python main.py method.algorithm='SOCM','UW_SOCM','UW_SOCM_diag','UW_SOCM_diag_2B','SOCM_const_M','SOCM_adjoint','work_adjoint','discrete_adjoint','cross_entropy','log-variance','moment','variance','reinf','reinf_fr','SOCM_cost','SOCM_cost_diag','SOCM_cost_diag_2B','reinf_unadj','SOCM_work','SOCM_work_diag','SOCM_work_diag_2B' method.lmbd=1.0 method.gamma=6.0 method.setting='double_well' method.d=10 method.num_steps=200 method.delta_t_optimal=0.001 method.delta_x_optimal=0.001 method.n_samples_control=65536 method.scaling_factor_M=0.1 optim.M_lr=3e-4 optim.batch_size=128 method.num_iterations=80000 method.seed=0 method.save_every=10 -m`

To get the plots:

`python plots.py method.lmbd=1.0 method.gamma=6.0 method.setting='double_well' method.d=10 method.num_steps=200 method.delta_t_optimal=0.001 method.delta_x_optimal=0.001 method.n_samples_control=65536 method.scaling_factor_M=0.1 optim.M_lr=3e-4 optim.batch_size=128 method.num_iterations=80000 method.seed=0 method.plot_number=1`

## Quadratic OU no state cost
To run the algorithms:

`python main.py method.algorithm='SOCM','UW_SOCM','UW_SOCM_diag','UW_SOCM_diag_2B','SOCM_const_M','SOCM_adjoint','work_adjoint','discrete_adjoint','cross_entropy','log-variance','moment','variance','reinf','reinf_fr','SOCM_cost','SOCM_cost_diag','SOCM_cost_diag_2B','reinf_unadj','SOCM_work','SOCM_work_diag','SOCM_work_diag_2B' method.lmbd=1.0 method.setting='OU_quadratic_no_state_cost' method.gamma=2.0 method.scaling_factor_M=0.1 optim.M_lr=3e-4 optim.batch_size=128 method.num_iterations=60000 method.save_every=10 -m`

To get the plots:

`python plots.py method.lmbd=1.0 method.setting='OU_quadratic_no_state_cost' method.gamma=2.0 method.scaling_factor_M=0.1 optim.M_lr=3e-4 optim.batch_size=128 method.plot_number=1`

## Sampling Funnel
<!-- `python main.py method.algorithm='discrete_adjoint' method.d=10 method.T=5.0 method.num_steps=100 method.lmbd=1.0 method.setting='sampling_funnel' method.gamma=2.0 method.scaling_factor_M=0.1 optim.M_lr=3e-4 optim.batch_size=128 method.num_iterations=60000 method.save_every=10` -->
To run the algorithms:

`python main.py method.algorithm='SOCM','UW_SOCM','UW_SOCM_diag','UW_SOCM_diag_2B','SOCM_const_M','SOCM_adjoint','work_adjoint','discrete_adjoint','cross_entropy','log-variance','moment','variance','reinf','reinf_fr','SOCM_cost','SOCM_cost_diag','SOCM_cost_diag_2B','reinf_unadj','SOCM_work','SOCM_work_diag','SOCM_work_diag_2B' method.d=10 method.T=5.0 method.num_steps=100 method.lmbd=1.0 method.setting='sampling_funnel' method.gamma=2.0 method.scaling_factor_M=0.1 optim.M_lr=3e-4 optim.batch_size=128 method.num_iterations=60000 method.save_every=10 -m`

To get the plots:

`python plots.py method.d=10 method.T=5.0 method.num_steps=100 method.lmbd=1.0 method.setting='sampling_funnel' method.gamma=2.0 method.scaling_factor_M=0.1 optim.M_lr=3e-4 optim.batch_size=128 method.save_every=10 method.plot_number=9`

## Sampling Cox
`python main.py method.algorithm='UW_SOCM' method.d=1600 method.lmbd=1.0 method.T=5.0 method.num_steps=100 method.setting='sampling_cox' method.gamma=2.0 method.scaling_factor_M=0.1 optim.M_lr=3e-4 optim.batch_size=128 method.num_iterations=60000 method.efficient_memory=True method.output_matrix=False`

## Sampling MG
`python main.py method.algorithm='discrete_adjoint' method.d=2 method.lmbd=1.0 method.T=5.0 method.num_steps=100 method.setting='sampling_MG' method.gamma=2.0 method.scaling_factor_M=0.1 optim.M_lr=3e-4 optim.batch_size=128 method.num_iterations=60000`

`python plots.py method.d=2 method.T=5.0 method.num_steps=100 method.lmbd=1.0 method.setting='sampling_MG' method.gamma=2.0 method.scaling_factor_M=0.1 optim.M_lr=3e-4 optim.batch_size=128 method.save_every=10 method.plot_number=9`
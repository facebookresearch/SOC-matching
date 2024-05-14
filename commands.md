## Quadratic OU easy
To run the algorithms:

`python main.py method.algorithm='SOCM','UW_SOCM','UW_SOCM_diag','UW_SOCM_diag_2B','SOCM_const_M','SOCM_adjoint','continuous_adjoint','discrete_adjoint','cross_entropy','log-variance','moment','variance','reinf','reinf_fr','SOCM_cost','SOCM_cost_diag','SOCM_cost_diag_2B','reinf_unadj','SOCM_work','SOCM_work_diag','SOCM_work_diag_2B' method.lmbd=1.0 method.setting='OU_quadratic_easy' method.gamma=2.0 method.scaling_factor_M=0.1 optim.M_lr=1e-3 optim.batch_size=128 method.num_iterations=60000 -m`

To get the plots:

`python plots.py method.lmbd=1.0 method.setting='OU_quadratic_easy' method.gamma=2.0 method.scaling_factor_M=0.1 optim.M_lr=1e-3 optim.batch_size=128 method.plot_number=1`

## Quadratic OU hard, no warm start
To run the algorithms:

`python main.py method.algorithm='SOCM','UW_SOCM','UW_SOCM_diag','UW_SOCM_diag_2B','SOCM_const_M','SOCM_adjoint','continuous_adjoint','discrete_adjoint','cross_entropy','log-variance','moment','variance','reinf','reinf_fr','SOCM_cost','SOCM_cost_diag','SOCM_cost_diag_2B','reinf_unadj','SOCM_work','SOCM_work_diag','SOCM_work_diag_2B' method.lmbd=1.0 method.setting='OU_quadratic_hard' method.gamma=2.0 method.scaling_factor_M=0.1 optim.M_lr=1e-3 optim.batch_size=128 method.num_iterations=80000 -m`

To get the plots:

`python plots.py method.lmbd=1.0 method.setting='OU_quadratic_hard' method.gamma=2.0 method.scaling_factor_M=0.1 optim.M_lr=1e-3 optim.batch_size=128 method.num_iterations=80000 method.plot_number=1`

## Linear OU
To run the algorithms:

`python main.py method.algorithm='SOCM','UW_SOCM','UW_SOCM_diag','UW_SOCM_diag_2B','SOCM_const_M','SOCM_adjoint','continuous_adjoint','discrete_adjoint','cross_entropy','log-variance','moment','variance','reinf','reinf_fr','SOCM_cost','SOCM_cost_diag','SOCM_cost_diag_2B','reinf_unadj','SOCM_work','SOCM_work_diag','SOCM_work_diag_2B' method.d=10 method.lmbd=1.0 method.gamma=2.0 method.setting='OU_linear' method.num_steps=100 method.scaling_factor_M=0.1 optim.M_lr=1e-3 optim.batch_size=64 method.num_iterations=60000 -m`

To get the plots:

`python plots.py method.d=10 method.lmbd=1.0 method.gamma=2.0 method.setting='OU_linear' method.num_steps=100 method.scaling_factor_M=0.1 optim.M_lr=1e-3 optim.batch_size=64 method.num_iterations=60000 method.plot_number=1`

## Double Well
To run the algorithms:

`python main.py method.algorithm='SOCM','UW_SOCM','UW_SOCM_diag','UW_SOCM_diag_2B','SOCM_const_M','SOCM_adjoint','continuous_adjoint','discrete_adjoint','cross_entropy','log-variance','moment','variance','reinf','reinf_fr','SOCM_cost','SOCM_cost_diag','SOCM_cost_diag_2B','reinf_unadj','SOCM_work','SOCM_work_diag','SOCM_work_diag_2B' method.lmbd=1.0 method.gamma=6.0 method.setting='double_well' method.d=10 method.num_steps=200 method.delta_t_optimal=0.001 method.delta_x_optimal=0.001 method.n_samples_control=65536 method.scaling_factor_M=0.1 optim.M_lr=1e-3 optim.batch_size=128 method.num_iterations=80000 method.seed=0 -m`

To get the plots:

`python plots.py method.lmbd=1.0 method.gamma=6.0 method.setting='double_well' method.d=10 method.num_steps=200 method.delta_t_optimal=0.001 method.delta_x_optimal=0.001 method.n_samples_control=65536 method.scaling_factor_M=0.1 optim.M_lr=1e-3 optim.batch_size=128 method.num_iterations=80000 method.seed=0 method.plot_number=1`

## Quadratic OU no state cost
To run the algorithms:

`python main.py method.algorithm='SOCM','UW_SOCM','UW_SOCM_diag','UW_SOCM_diag_2B','SOCM_const_M','SOCM_adjoint','continuous_adjoint','discrete_adjoint','cross_entropy','log-variance','moment','variance','reinf','reinf_fr','SOCM_cost','SOCM_cost_diag','SOCM_cost_diag_2B','reinf_unadj','SOCM_work','SOCM_work_diag','SOCM_work_diag_2B' method.lmbd=1.0 method.setting='OU_quadratic_no_state_cost' method.gamma=2.0 method.scaling_factor_M=0.1 optim.M_lr=1e-3 optim.batch_size=128 method.num_iterations=60000 -m`

To get the plots:

`python plots.py method.lmbd=1.0 method.setting='OU_quadratic_no_state_cost' method.gamma=2.0 method.scaling_factor_M=0.1 optim.M_lr=1e-3 optim.batch_size=128 method.plot_number=1`
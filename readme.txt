Custom code for generating and analyzing the data envolved in this paper are presented.
Minimal datasets for generating the main figures are also presented.

The RNN codes are based on Python 3.6.5 and Tensorflow 1.8.0
Some data analysis codes are based on MATLAB-2017a

A typical training of the RNN usually takes a few minutes on a desktop.
The Hill-function based enumerative search related to fig.3 may take hours to complete.

The source codes and sample results are organized following the main figures.




# fig1-2_adaptation #
-----------------------------
'rnn_models.py' contains the definition of the RNN model. 
'target_response_curves.py' defines the fitting targets used in figs. 1-2 and fig. S1. 

'fig1_1_train.py' train the model, with the output folder name provided as the input argument '--output_name 1'. And the results are stored in the output folder '1'.
'1/loss.csv' records the training error.
'1/train' stores some intermediate outputs during training. 
'1/savenet' contains the Tensorflow checkpoint files.
'1/paras' contains the NN parameters (weights and bias) as .csv files. 

'fig1_2_test.py' generates test output trajectories under 20 different stimuli strengths.
These test results are documented in the 'trajs' folder.
The output file 'trajs/run1_traj_g*.csv' contains a [20,T] matrix, with each row represents one of the 20 different input cases.

'fig1c.m' reads the NN parameters from '1/paras', re-implements the RNN model, and plots fig.1c.



'fig2_1_train.py' trains the RNN model with predefined network connectivity.
An input file (e.g. '1/links.csv') that defines the allowed regulatory links should be provided in the output directory (in this case '1/').
If the output directory does not exist, all regulatory links are allowed by default.

'fig2_2_link_knockout.py' is used to find an effective regulation network with the "link mutation" technique.
It also generate a new 'link.csv' file with the currently "most irrelavant" link being removed.
For example, When run on the output folder '--output_name $n', this program generates a new "link" file at '$n+1/links.csv' where the most irrelavent link is removed.
The input arguments are: 
'--output_name' specifies the directory of the trained NN.
'--lambda_factor' defines the discount factor for link mutaion test (see equation 3 in the main text). 
'--keep_input_links' switch between two mode for determining "the least relevant link". If it equals to 1, the links from the inpiut stimuli "I" to the other nodes "gs" will never be removed, no matter how weak it appears to be. This function will not be activated if this argument is set to 0 (as default).

To generate sparse network trajectories like fig.2c, one need to run 'fig2_1_train.py' and 'fig2_2_link_knockout.py' iteratively. eg:
  python fig2_1_train.py --output_name 1
  python fig2_2_link_knockout.py --output_name 1
  python fig2_1_train.py --output_name 2
  python fig2_2_link_knockout.py --output_name 2
  python fig2_1_train.py --output_name 3
  python fig2_2_link_knockout.py --output_name 3
  ...

The results in folders 11-16 corresponds to fig.2c; and results 21-25 corresponds to fig.2d.

'fig2cd.m' reads the NN parameters from 'trajs' and plots fig.2c and d.




# fig3-4 controlled_oscillation #
-----------------------------
RNN model used for fig.3b-f has the same structure as fig.2, with all links allowed in training.
The codes are in parallel with those for the adaptation case basically.
RNN models for fig.4 has the same structure as fig.2c-d.
The results in folders 61-68 corresponds to fig.4a-b




# fig5_gap_gene_patterning #
-----------------------------
'Data_frame' contains a single frame of the 1-dimensional gap gene pattern from the FlyEX database.
Colums 1 to 5 represents gap genes hb, Kr, kni, gt, and tll. Note that the profile of tll is not used in this paper. 

Folders '1-10' show some typical training repeats.
For example, '1/train' folder contain fittings to the desired pattern in the intermediate training steps.
The effective regulation network obtaned by "link mutation" ('fig5_2_link_mutation.py') is stored in 'nets'.

'fig5_3_test.py' performs tests on trained RNN model, and writes results to '*/test'.
For exampel, '1/test' contains model fitting (and prediction) results on "wild-type" (and on several mutants).

'fig5b.m' plots the WT fitting results. 
'fig5e.m' plots model predictions on Kruppel mutant. 

Note that training is repeated for 50 times, among which 10 repeats failed to converge thus excluded for analysis or prediction. 
To keep file size small, only the first 10 training repeats are included here. Among them, trails 3,5,9 has relatively large training error thus being excluded.




# fig6_reverse_engineering_CA #
-----------------------------
'fig6_0_rand-sample_HF.py' implements the Hill-Function based Cellular-Automatum model. 
This program samples network topology and HF parameters randomly, and searches for HF-based CA model that displays non-uniform spatial and temporal patterns. 
Some searching results are shown in 'CA_ground_truth_models/imgs', form which we select arbitrarily 25 models as "ground truth" for further study. 
Note that a Gaussian noise is added when simulating the CA dynamics, therefore although the models should have reflection symmetry, the spatial patterns may by unsymmetric.
Those CA models that displays "more symmetrical" patterns should be more robust against noise, and are preferred when selecting "ground truth" models.

Network topology and paramenters of the 25 selected ground truth models are stored in 'CA_ground_truth_models/para/paras_ground_truth.csv'. 
This file contains a 25x600 matrix, with each row representing a ground truth CA model. 
For each model, there are 200 possible regulatory links; and each link has its own sign and parameter. 
The first 200 colums represents the 'b' term for this link (it is used only when the link is an activating one) and is all set to 1 in this study.
The second 200 colums are the 'K' term, which is the Michaelis constant in the HF formula (see Methods). 
The last 200 colums represents the ground truth network topology. +1 stands for an activating link, -1 stands for an inhibiting one, and 0 stands for non-exist.

'fig6_1_train.py' trains RNN model to simulate the CA dynamics.
Trained models are stored in 'trained_RNN_models/model_*/', where * means the number of ground truth CA model.

'fig6_2_link_knockout.py' is used to reveal the effective regulation network of the trained RNN. 
The delta_ij term introduced in equation 2 is computed, with the lambda factor set to 0. 
This procedure yeilds a 20x10 regulation matrix, stored as 'trained_RNN_models/nets/model_*.csv'.

'ROC_of_RNN.m' computs the ROC curve of the NN-predicted activating and/or inhibiting links for all the ground truth CA models (fig.6h and fig.S6).
To keep file size small, only the training and testing results on CAs #7 and #24 are included. 

%models = 'ecctn_1'
%dirs = '/Work19/2019/nabil/ecctn_1/exp/ecctn_1/'

addpath(genpath('./matlab/PESQ/MATLAB/'));
addpath(genpath('./matlab/SRMR/MATLAB/'));
addpath(genpath('./matlab/STOI/MATLAB/'));

disp('Computing PESQ')
pesq_main(models, dirs);
disp('PESQ score saved!')

disp('Computing SRMR')
srmr_main(models, dirs);
disp('SRMR score saved!')

disp('Computing STOI')
stoi_main(models, dirs);
disp('STOI score saved!')
disp('Evaluation complete!')

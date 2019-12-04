% The following code is used to train 2000-level adaboost model using 
% 'train.csv' as the training data and 'test.csv'(unlabeled) as the testing
% data. Although I used class labels of -1 and 1 instead of 0 and 1 in 
% order to use 'sign' function to draw final prediction of an observation,
% I convert the final decision back to 0 and 1 for submission on Kaggle.
% You can copy the final result from 'final_estimation_adaboost' variable
% to the csv file for submission to get your results evaluated by Kaggle.

clear,close all;clc

% import 'train.csv' file
opt_training_data = detectImportOptions('train.csv','ReadVariableNames',true);
training_data = readtable('train.csv',opt_training_data);

% use class labels of -1 and 1 instead of 0 and 1 in order to use 'sign' 
% function to draw final prediction of an observation 
training_data.eyeDetection(training_data.eyeDetection == 0) = -1;

% import 'test.csv'
opt_testing_data = detectImportOptions('test.csv','ReadVariableNames',true);
testing_data = readtable('test.csv',opt_testing_data);

% find the number of training and testing observations
testing_data_num = height(testing_data);
training_data_num = height(training_data);

% Initialization

% this variable contains training set for each iteration, it will be
% updated at the beginning of each iteration
training_data_adaboost = training_data;

% this variables contains the index of every training observation in the
% original entire training set, it is used to update weight of every
% observation
initial_index = zeros(training_data_num,1);

% initial weight
weight = 1/training_data_num * ones(training_data_num,1);

% number of levels of adaboost
num_of_levels = 2000;

% the same as 'alpha' in class notes
alpha = zeros(1,num_of_levels);

% initializaiton of variable used to save prediction results of testing set
estimated_class_adaboost = zeros(testing_data_num,num_of_levels);

% number of samples need to be selected in each bootstrap process
num_of_samples = training_data_num;

% a variable used to record weight of bootstrap in every iteration
weight_record = zeros(training_data_num,num_of_levels);

% a variable used to record 'error' in every level as shown in class note
error = zeros(1,num_of_levels);

% training process of 'num_of_levels'-level adaboost, you can specify
% the number of levels of adaboost above
for iteration = 1:num_of_levels
    % this variable is used to record weight of each observation in every iteration, and it's only used for debugging
    weight_record(:,iteration) = weight; % used to record weight of each observation in each iteration
    
    % sampling from training_data with replacement according to weight
    [training_data_adaboost,initial_index] = datasample(training_data,num_of_samples,'Weights',weight);
    
    % fit a classifier
    tree_adaboost  = fitctree(training_data_adaboost,'eyeDetection','MaxNumSplits', 500,'SplitCriterion','deviance');
    
    % classify testing data
    estimated_class_adaboost(:,iteration) = predict(tree_adaboost,testing_data);    
    
    % apply the tree to training data
    estimated_class_training = predict(tree_adaboost,training_data);
    
    % find erroneously classified data
    missclassified_training_data = (estimated_class_training ~= training_data.eyeDetection);

    % culculate 'error', according to notes
    error(iteration) = sum(weight(missclassified_training_data));
    
    % calcuate 'alpha', according to notes
    alpha(iteration) = log((1-error(iteration))/error(iteration));
    
    % reassign weight
    weight(missclassified_training_data) = weight(missclassified_training_data) * exp(alpha(iteration));
    weight = weight ./ sum(weight); % weight normalizaiton
end

% calculate weighted sum of classification results of appointed levels to draw the
% final estimation
final_estimation_adaboost = sign(sum(estimated_class_adaboost.*alpha,2));
% convert the final decision back to 0 and 1 for submission on Kaggle
final_estimation_adaboost(final_estimation_adaboost<0) = 0;
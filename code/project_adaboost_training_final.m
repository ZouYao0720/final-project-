% The following code is the training process of adaboost classifier. I
% leave first 10% of entire data provided in 'train.csv' as test data

clear,close all;clc

% import train.csv file
opt_training_data = detectImportOptions('train.csv','ReadVariableNames',true);
entire_data = readtable('train.csv',opt_training_data);

% use class labels of -1 and 1 instead of 0 and 1 in order to use 'sign' 
% function to draw final prediction of an observation 
entire_data.eyeDetection(entire_data.eyeDetection == 0) = -1;

% split entire data set into training and testing data
entire_data_height = height(entire_data);
testing_fraction = 0.1;
training_fraction = 1 - testing_fraction;
testing_data_num = testing_fraction * entire_data_height;
training_data_num = training_fraction * entire_data_height;
testing_data = entire_data(1:testing_data_num,:);
training_data = entire_data(testing_data_num+1:end,:);

% Initialization

% this variable contains training set for each iteration, it will be
% updated at the beginning of each iteration
training_data_adaboost = training_data;

% this variables contains the index of every training observation in the (after bootstraping)
% original training set, it is used to update weight of every observation
initial_index = zeros(training_data_num,1);

% initial weight
weight = 1/training_data_num * ones(training_data_num,1);

% number of levels of adaboost
num_of_levels = 100;

% the same as 'alpha' in notes
alpha = zeros(1,num_of_levels);

% initializaiton of variable used to save prediction results of testing set
estimated_class_adaboost = zeros(testing_data_num,num_of_levels);

% the number of samples need to be selected in each bootstrap process
num_of_samples = training_data_num;

% a variable used to record weight of bootstrap in every iteration
weight_record = zeros(training_data_num,num_of_levels);

% a variable used to record 'error' in every level as shown in class note
error = zeros(1,num_of_levels);

% this variable is used to contain the error rate of adaboost with 1 to
% 10000 levels on the test data
testing_error_rate = zeros(num_of_levels,1);

% training process of 'num_of_levels'-level adaboost, you can specify
% the number of levels of adaboost above
for iteration = 1:num_of_levels
    % this variable is used to record weight of each observation in every iteration, and it's only used for debugging
    weight_record(:,iteration) = weight; 
    
    % sampling from training_data with replacement according to weight (bootstraping process)
    [training_data_adaboost,initial_index] = datasample(training_data,num_of_samples,'Weights',weight);
    
    % fit a classifier
    tree_adaboost  = fitctree(training_data_adaboost,'eyeDetection','SplitCriterion','deviance');
    
    % classify testing data
    estimated_class_adaboost(:,iteration) = predict(tree_adaboost,testing_data);    
    
    % apply the tree to training data
    estimated_class_training = predict(tree_adaboost,training_data);
    
    % find erroneously classified data
    missclassified_training_data = (estimated_class_training ~= training_data.eyeDetection);
    
    % updata the error rate of adaboost with current level number on the
    % testing data
    testing_error_rate(iteration,1) = sum(estimated_class_adaboost(:,iteration) ~= testing_data.eyeDetection)/testing_data_num;
    
    % culculate 'error', according to notes
    error(iteration) = sum(weight(missclassified_training_data));
    
    % calcuate 'alpha', according to notes
    alpha(iteration) = log((1-error(iteration))/error(iteration));
    
    % reassign weight
    weight(missclassified_training_data) = weight(missclassified_training_data) * exp(alpha(iteration));
    weight = weight ./ sum(weight); % weight normalizaiton
end

% calculate weighted sum of classification results of 'num_of_levels' levels to draw the
% final estimation
final_estimation_adaboost = sign(sum(estimated_class_adaboost.*alpha,2));

% As 'plotconfusion' must have categorical input arguments, I convert the
% testing data label and their final prediction to catelogical variables
final_estimation_adaboost = categorical(final_estimation_adaboost);
cat_testing_data.eyeDetection = categorical(testing_data.eyeDetection);
plotconfusion(cat_testing_data.eyeDetection,final_estimation_adaboost);
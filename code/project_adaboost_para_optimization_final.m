% The following code is used to optimize a hyperparameter, the number of
% levels of adaboost. The final result is demonstrated by a figure named 
% 'Error of adaboost with different number of levels', which shows the
% relation between the error of adaboost and the number of its levels

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

% this variables contains the index of every training observation (after bootstraping) in the
% original training set, it is used to update weight of every observation
initial_index = zeros(training_data_num,1);

% initial weight
weight = 1/training_data_num * ones(training_data_num,1);

i = 1; % initialization of loop counting
num_of_iterations = 10000; % adaboost executes from 1 up to 10000 levels
% this variable is used to contain the error rate 
error_rate = zeros(1,num_of_iterations);
alpha = zeros(1,num_of_iterations); % the same as 'alpha' in class notes

% initializaiton of variable used to save prediction results of testing set
estimated_class_adaboost = zeros(testing_data_num,num_of_iterations);

% the number of samples need to be selected in each bootstrap process
num_of_samples = training_data_num;

% a variable used to record weight of bootstrap in every iteration
weight_record = zeros(training_data_num,num_of_iterations);

% a variable used to record 'error' in every level as shown in class note
error = zeros(1,num_of_iterations);

% this variable is used to contain the error rate of adaboost with 1 to
% 10000 levels on the test data
testing_error_rate = zeros(num_of_iterations,1);

% training process of adaboost with 1 to 10000 levels
for iteration = 1:num_of_iterations
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
    
    % the final prediction of adaboost with current number of levels on
    % testing data
    final_estimation_adaboost = sign(sum(estimated_class_adaboost.*alpha,2));
    
    % calculate the error rate of adaboost with current number of
    % levels on testing data
    error_rate(i) = sum(final_estimation_adaboost ~= testing_data.eyeDetection)/testing_data_num;
    i = i+1; % update loop counting
end

% demonstration of results
figure
final_estimation_adaboost = categorical(final_estimation_adaboost);
cat_testing_data.eyeDetection = categorical(testing_data.eyeDetection);
plotconfusion(cat_testing_data.eyeDetection,final_estimation_adaboost);

figure
semilogx([1:num_of_iterations],error_rate)
set(gca,'fontsize',30);
title('Error of adaboost with different number of levels','FontSize',30)
xlabel('Number of levels of adaboost','FontSize',30)
ylabel('Error rate of adaboost','FontSize',30)
grid on

figure
semilogx([1:num_of_iterations],alpha)
set(gca,'fontsize',30);
title('\alpha of each level of adaboost','FontSize',30)
xlabel('levels of adaboost','FontSize',30)
ylabel('\alpha','FontSize',30)
grid on

figure
semilogx([1:num_of_iterations],testing_error_rate)
set(gca,'fontsize',30);
title('Error rate of each single level of adaboost on test data','FontSize',30)
xlabel('levels of adaboost','FontSize',30)
ylabel('error rate of applying 1 specific level to test data','FontSize',20)
grid on
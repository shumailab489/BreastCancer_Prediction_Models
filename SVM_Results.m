clear; close all; clc;

%% Load and preprocess your dataset
data = readtable("D:\Research Work\BC data\data.csv");

% Assume "diagnosis" is the target variable
Y = grp2idx(data.diagnosis)
X = table2array(data(:, 3:end))

%% Split the data into training and testing sets
 rng(42); % Set a seed for reproducibility
cv = cvpartition(Y, 'HoldOut', 0.2); % 80% train, 20% test
X_train = X(cv.training,:);
y_train = Y(cv.training,:);
X_test = X(cv.test,:);
y_test = Y(cv.test,:);

%% Feature scaling (Standardization)
X_train = zscore(X_train);
X_test = zscore(X_test);

%% Hyperparameter tuning using grid search
svm_model = fitcsvm(X_train, y_train, 'KernelFunction', 'rbf', 'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', 'expected-improvement-plus', 'ShowPlots', true));

%% Final test with the test set
y_pred = predict(svm_model, X_test);
accuracy = sum(y_pred == y_test) / numel(y_test) * 100;

fprintf('Test accuracy: %.2f%%\n', accuracy);

%% 3D Scatter Plot
figure;
scatter3(X(:, 1), X(:, 2), X(:, 3), 10, Y, 'filled', 'MarkerEdgeColor', 'r');
xlabel('Variable 1');
ylabel('Variable 2');
zlabel('Variable 3');
hold on



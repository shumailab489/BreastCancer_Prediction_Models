clc
clear all

% Load the data
data = readtable("D:\Research Work\BC data\data2.csv");

X = data(:, 1:end-1); % Features
y = data.Status; % Target variable

rng(42); % For reproducibility
cv = cvpartition(y, 'HoldOut', 0.2); % 80% training, 20% testing
X_train = X(training(cv), :);
y_train = y(training(cv), :);
X_test = X(test(cv), :);
y_test = y(test(cv), :);

Mdl = TreeBagger(100, X_train, y_train, 'Method', 'classification');

% Predictions
y_pred = predict(Mdl, X_test);

accuracy = sum(strcmp(y_pred, y_test)) / numel(y_test);
disp(['Accuracy on test data: ' num2str(accuracy)]);

train_acc = sum(strcmp(predict(Mdl, X_train), y_train)) / numel(y_train);


% Accuracy Plot
figure;
bar([1, 2], [train_acc, accuracy], 'BarWidth', 0.4, 'FaceColor', 'b');
xticks([1, 2]);
xticklabels({'Training', 'Test'});
ylabel('Accuracy');
title('Classification Accuracy');

% Confusion Matrix with Title
figure;
confusionchart(y_test, y_pred, 'Title', 'Confusion Matrix');
C = confusionmat(y_test, y_pred);
disp('Confusion Matrix:');
disp(C);


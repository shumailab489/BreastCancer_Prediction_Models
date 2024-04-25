clc
clear all

% Load the data
data = readtable("D:\Research Work\BC data\data.csv");

X = data(:, 3:end); % Features
y = data.diagnosis; % Target variable

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

% Scatter Plot (Test Data)
figure;
gscatter(X_test{:,1}, X_test{:,2}, y_test, 'br', '*o', 8);
xlabel('Feature 1');
ylabel('Feature 2');
title('Scatter Plot of Test Data');
legend('Class0', 'Class1');
grid on

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


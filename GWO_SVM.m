 clc
 clear all
data2 = readtable("D:\Research Work\BC data\data.csv");

% Split data into training and test sets
rng(123); % For reproducibility
cv = cvpartition(size(data2, 1), 'Holdout', 0.2);
trainData = data2(training(cv), :);
testData = data2(test(cv), :);

% Feature scaling
featureColumns = trainData.Properties.VariableNames(3:end);
trainData{:, featureColumns} = zscore(trainData{:, featureColumns});
testData{:, featureColumns} = zscore(testData{:, featureColumns});

% Grey Wolf Optimizer (GWO) hyperparameters
maxIter = 50; % Number of iterations
nPop = 10;     % Number of wolves

% Initialize GW positions
positions = randn(nPop, length(featureColumns) - 2); % Exclude id and diagnosis columns
convergenceCurve = zeros(maxIter, 1);

for iter = 1:maxIter
    % Train SVM with GWO-optimized parameters
    accuracies = zeros(nPop, 1);   
    for i = 1:nPop
        C = 2^positions(i, 1);
        gamma = 2^positions(i, 2);
        
        % Adjust the C value if needed
        if C < 1e-3
            C = 1e-3;
        end
        
        svmModel = fitcsvm(trainData{:, featureColumns(3:end)}, trainData.diagnosis, ...
            'BoxConstraint', C, 'KernelFunction', 'RBF', 'KernelScale', gamma);
        
        % Predict on test data
        predictions = predict(svmModel, testData{:, featureColumns(3:end)});
        accuracies(i) = sum(strcmp(predictions, testData.diagnosis)) / numel(predictions);
    end
    
    % Update GWO positions based on fitness (accuracy)
    [~, alphaIdx] = max(accuracies);
    accuracies(alphaIdx) = -Inf; % Remove the best solution from consideration
    
    [~, betaIdx] = max(accuracies);
    accuracies(betaIdx) = -Inf;  % Remove the second best solution
    
    [~, deltaIdx] = max(accuracies);
    
    alpha = positions(alphaIdx, :);
    beta = positions(betaIdx, :);
    delta = positions(deltaIdx, :);
    
    a = 2 - iter * ((2) / maxIter);
    
    for i = 1:nPop
        for j = 1:size(positions, 2)
            r1 = rand(); % Random numbers between 0 and 1
            r2 = rand();
            A1 = 2 * a * r1 - a;
            C1 = 2 * r2;
            
            D_alpha = abs(C1 * alpha(j) - positions(i, j));
            X1 = alpha(j) - A1 * D_alpha;
            
            positions(i, j) = X1;
        end
    end
    
    % Record the best accuracy for convergence analysis
    convergenceCurve(iter) = max(accuracies);
end

% Select the best solution (position) and train SVM on the entire training set
bestPosition = positions(alphaIdx, :);
bestC = 2^bestPosition(1);
bestGamma = 2^bestPosition(2);

finalSvmModel = fitcsvm(trainData{:, featureColumns(3:end)}, trainData.diagnosis, ...
    'BoxConstraint', bestC, 'KernelFunction', 'RBF', 'KernelScale', bestGamma);

% Predict on test data using the final trained model
finalPredictions = predict(finalSvmModel, testData{:, featureColumns(3:end)});
finalAccuracy = sum(strcmp(predictions, testData.diagnosis)) / numel(predictions);
fprintf('Final Test Accuracy: %.2f%%\n', finalAccuracy * 100);
% Plot the convergence curve
figure;
plot(convergenceCurve,'r','LineWidth',1);
xlabel('Iteration');
ylabel('Accuracy');
title('Convergence of GWO-SVM');
grid on
% % Scatter Plot for Training Data
% figure;
% gscatter(trainData{:, 'Var3'}, trainData{:, 'Var4'}, trainData.Var2, 'br', 'o*', 8);
% xlabel('Mean Radius');
% ylabel('Mean Texture');
% title('Scatter Plot of Training Data');
% legend('Class0', 'Class1');
% grid on
% 
% % Confusion Matrix for Training Data
% trainPredictions = predict(finalSvmModel, trainData{:, featureColumns(3:end)});
% trainAccuracy = sum(strcmp(trainPredictions, trainData.Var2)) / numel(trainPredictions);
% figure;
% confusionchart(trainData.Var2, trainPredictions);
% trainConfusion = confusionmat(trainData.Var2, trainPredictions);
% disp('Confusion Matrix for Training Data:');
% disp(trainConfusion);
% 
% % Confusion Matrix for Test Data
% figure;
% 
% testPredictions = finalPredictions; % Use the predictions from the final model
% testAccuracy = sum(strcmp(testPredictions, testData.Var2)) / numel(testPredictions);
%  confusionchart(testData.Var2, testPredictions);
% testConfusion = confusionmat(testData.Var2, testPredictions);
% disp('Confusion Matrix for Test Data:');
% disp(testConfusion);
   
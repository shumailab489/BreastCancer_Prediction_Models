function [trainedClassifier, validationAccuracy] = Fine_KNN_RelifF(trainingData)
inputTable = trainingData;
predictorNames = {'id', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concavePoints_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concavePoints_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concavePoints_worst', 'symmetry_worst', 'fractal_dimension_worst'};
predictors = inputTable(:, predictorNames);
response = inputTable.diagnosis;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% Feature Ranking and Selection
% Replace Inf/-Inf values with NaN to prepare data for normalization
predictors = standardizeMissing(predictors, {Inf, -Inf});
% Normalize data for feature ranking
predictorMatrix = normalize(predictors, "DataVariable", ~isCategoricalPredictor);
isAllPredictorsCategorical = all(isCategoricalPredictor);
if isAllPredictorsCategorical
    newPredictorMatrix = zeros(size(predictorMatrix));
    for i = 1:size(predictorMatrix, 2)
        newPredictorMatrix(:,i) = grp2idx(predictorMatrix{:,i});
    end
    predictorMatrix = newPredictorMatrix;
else
    predictorMatrix = table2array(predictorMatrix);
    responseVector = response;
end

% Rank features using ReliefF algorithm
featureIndex = relieff(...
    predictorMatrix, ...
    responseVector, ...
    10);
numFeaturesToKeep = 10;
includedPredictorNames = predictors.Properties.VariableNames(featureIndex(1:numFeaturesToKeep));
predictors = predictors(:,includedPredictorNames);
isCategoricalPredictor = isCategoricalPredictor(featureIndex(1:numFeaturesToKeep));

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
classificationKNN = fitcknn(...
    predictors, ...
    response, ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 1, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', categorical({'B'; 'M'}));

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
featureSelectionFcn = @(x) x(:,includedPredictorNames);
knnPredictFcn = @(x) predict(classificationKNN, x);
trainedClassifier.predictFcn = @(x) knnPredictFcn(featureSelectionFcn(predictorExtractionFcn(x)));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = {'area_mean', 'area_se', 'area_worst', 'compactness_mean', 'compactness_se', 'compactness_worst', 'concavePoints_mean', 'concavePoints_se', 'concavePoints_worst', 'concavity_mean', 'concavity_se', 'concavity_worst', 'fractal_dimension_mean', 'fractal_dimension_se', 'fractal_dimension_worst', 'id', 'perimeter_mean', 'perimeter_se', 'perimeter_worst', 'radius_mean', 'radius_se', 'radius_worst', 'smoothness_mean', 'smoothness_se', 'smoothness_worst', 'symmetry_mean', 'symmetry_se', 'symmetry_worst', 'texture_mean', 'texture_se', 'texture_worst'};
trainedClassifier.ClassificationKNN = classificationKNN;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2022b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictorNames = {'id', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concavePoints_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concavePoints_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concavePoints_worst', 'symmetry_worst', 'fractal_dimension_worst'};
predictors = inputTable(:, predictorNames);
response = inputTable.diagnosis;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% Set up holdout validation
cvp = cvpartition(response, 'Holdout', 0.2);
trainingPredictors = predictors(cvp.training, :);
trainingResponse = response(cvp.training, :);
trainingIsCategoricalPredictor = isCategoricalPredictor;

% Feature Ranking and Selection
% Replace Inf/-Inf values with NaN to prepare data for normalization
trainingPredictors = standardizeMissing(trainingPredictors, {Inf, -Inf});
% Normalize data for feature ranking
predictorMatrix = normalize(trainingPredictors, "DataVariable", ~trainingIsCategoricalPredictor);
isAllPredictorsCategorical = all(trainingIsCategoricalPredictor);
if isAllPredictorsCategorical
    newPredictorMatrix = zeros(size(predictorMatrix));
    for i = 1:size(predictorMatrix, 2)
        newPredictorMatrix(:,i) = grp2idx(predictorMatrix{:,i});
    end
    predictorMatrix = newPredictorMatrix;
else
    predictorMatrix = table2array(predictorMatrix);
    responseVector = trainingResponse;
end

% Rank features using ReliefF algorithm
featureIndex = relieff(...
    predictorMatrix, ...
    responseVector, ...
    10);
numFeaturesToKeep = 10;
includedPredictorNames = trainingPredictors.Properties.VariableNames(featureIndex(1:numFeaturesToKeep));
trainingPredictors = trainingPredictors(:,includedPredictorNames);
trainingIsCategoricalPredictor = trainingIsCategoricalPredictor(featureIndex(1:numFeaturesToKeep));

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
classificationKNN = fitcknn(...
    trainingPredictors, ...
    trainingResponse, ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 1, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', categorical({'B'; 'M'}));

% Create the result struct with predict function
featureSelectionFcn = @(x) x(:,includedPredictorNames);
knnPredictFcn = @(x) predict(classificationKNN, x);
validationPredictFcn = @(x) knnPredictFcn(featureSelectionFcn(x));

% Add additional fields to the result struct


% Compute validation predictions
validationPredictors = predictors(cvp.test, :);
validationResponse = response(cvp.test, :);
[validationPredictions, validationScores] = validationPredictFcn(validationPredictors);

% Compute validation accuracy
correctPredictions = (validationPredictions == validationResponse);
isMissing = ismissing(validationResponse);
correctPredictions = correctPredictions(~isMissing);
validationAccuracy = sum(correctPredictions)/length(correctPredictions);

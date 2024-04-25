
clc
clear all
rng(42)
%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 15);

% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["Age", "Race", "MaritalStatus", "TStage", "NStage", "sixthStage", "Grade", "AStage", "TumorSize", "EstrogenStatus", "ProgesteroneStatus", "RegionalNodeExamined", "ReginolNodePositive", "SurvivalMonths", "Status"];
opts.VariableTypes = ["double", "categorical", "categorical", "double", "double", "categorical", "categorical", "categorical", "double", "categorical", "categorical", "double", "double", "double", "categorical"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, ["Race", "MaritalStatus", "sixthStage", "Grade", "AStage", "EstrogenStatus", "ProgesteroneStatus", "Status"], "EmptyFieldRule", "auto");
opts = setvaropts(opts, ["TStage", "NStage"], "TrimNonNumeric", true);
opts = setvaropts(opts, ["TStage", "NStage"], "ThousandsSeparator", ",");

% Import the data
data2 = readtable("D:\Research Work\BC data\data2.csv", opts);

%% Preprocessing

% Convert categorical variables to numeric using one-hot encoding
encoded_race = dummyvar(categorical(data2.Race));
encoded_marital = dummyvar(categorical(data2.MaritalStatus));
encoded_sixthStage = dummyvar(categorical(data2.sixthStage));
encoded_grade = dummyvar(categorical(data2.Grade));
encoded_AStage = dummyvar(categorical(data2.AStage));
encoded_EstrogenStatus = dummyvar(categorical(data2.EstrogenStatus));
encoded_ProgesteroneStatus = dummyvar(categorical(data2.ProgesteroneStatus));
encoded_Status = dummyvar(categorical(data2.Status));

% Combine one-hot encoded variables
data2_encoded = [data2(:, {'Age', 'TStage', 'NStage', 'TumorSize', 'RegionalNodeExamined', 'ReginolNodePositive', 'SurvivalMonths'}), ...
                 array2table(encoded_race), array2table(encoded_marital), array2table(encoded_sixthStage), ...
                 array2table(encoded_grade), array2table(encoded_AStage), array2table(encoded_EstrogenStatus), ...
                 array2table(encoded_ProgesteroneStatus), array2table(encoded_Status)];

% Convert data2_encoded to an array
data2_array = table2array(data2_encoded);

% Normalize numeric variables
numeric_indices = [1, 4:9];
data2_array(:, numeric_indices) = normalize(data2_array(:, numeric_indices));

% Convert back to table
data2_encoded = array2table(data2_array, 'VariableNames', data2_encoded.Properties.VariableNames);

% Split the data into features and labels
X = data2_encoded(:,1:end-1); % Features
Y = data2_encoded(:,end);     % Labels

% Split data into training and testing sets (80% training, 20% testing)
cv = cvpartition(size(X,1),'HoldOut',0.2);
idxTrain = training(cv); % Logical index for training set
X_train = X{idxTrain,:};
Y_train = Y{idxTrain,:};
X_test = X{~idxTrain,:};
Y_test = Y{~idxTrain,:};

%% Grey Wolf Optimizer (GWO) with Support Vector Machine (SVM)

% GWO parameters
maxIter = 50; % Number of iterations
nPop = 10;    % Number of wolves

% Initialize positions of wolves
positions = rand(nPop, 2) * 2 - 1; % Random initialization between -1 and 1

% Initialize best position and best cost
best_position = zeros(1, 2);
best_cost = Inf;

% Initialize array to store costs at each iteration
costs = zeros(1, maxIter);

% GWO main loop
for iter = 1:maxIter
    for i = 1:nPop
        % Decode and update parameters
        C = 2^positions(i, 1);
        gamma = 2^positions(i, 2);
        
        % Train SVM model with current parameters
        svm_model = fitcsvm(X_train, Y_train, 'KernelFunction', 'rbf', 'BoxConstraint', C, 'KernelScale', gamma);
        
        % Evaluate the model
        Y_pred = predict(svm_model, X_test);
        accuracy = sum(Y_pred == Y_test) / numel(Y_test);
        cost = 1 - accuracy; % Cost function
        
        % Update best position and best cost
        if cost < best_cost
            best_cost = cost;
            best_position = positions(i,:);
        end
    end
    
    % Update positions using GWO
    a = 2 - iter * (2 / maxIter); % Linearly decreased from 2 to 0
    for i = 1:nPop
        r1 = rand();
        r2 = rand();
        A1 = 2 * a * r1 - a;
        C1 = 2 * r2;
        
        D_alpha = abs(C1 * best_position - positions(i,:)); % Distance to alpha
        X1 = best_position - A1 * D_alpha; % Update position using alpha
        
        r1 = rand();
        r2 = rand();
        A2 = 2 * a * r1 - a;
        C2 = 2 * r2;
        
        D_beta = abs(C2 * best_position - positions(i,:)); % Distance to beta
        X2 = best_position - A2 * D_beta; % Update position using beta
        
        r1 = rand();
        r2 = rand();
        A3 = 2 * a * r1 - a;
        C3 = 2 * r2;
        
        D_delta = abs(C3 * best_position - positions(i,:)); % Distance to delta
        X3 = best_position - A3 * D_delta; % Update position using delta
        
        % Update position of the current wolf
        positions(i,:) = (X1 + X2 + X3) / 3;
    end
    
    % Store the cost at the current iteration
    costs(iter) = accuracy; % Store accuracy instead of cost
end

%% Plot convergence curve
figure;
plot(1:maxIter, costs, 'r', 'LineWidth', 1); % Plot accuracy
xlabel('Iteration');
ylabel('Accuracy');
title('Convergence of GWO-SVM');
grid on;



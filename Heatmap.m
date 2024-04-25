clc
clear all

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 32);
% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concavePoints_mean", "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "concavePoints_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concavePoints_worst", "symmetry_worst", "fractal_dimension_worst"];
opts.VariableTypes = ["double", "categorical",  "double",  "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, "diagnosis", "EmptyFieldRule", "auto");

% Import the data
data = readtable("D:\Research Work\BC data\data.csv", opts);

%% Clear temporary variables
clear opts

% Create the heatmap
% Exclude "id" from correlation calculation
dataWithoutId = data(:, ~strcmp(data.Properties.VariableNames, 'id'));

% Convert "diagnosis" to numeric representation (M=1, B=0)
dataWithoutId.diagnosis = double(dataWithoutId.diagnosis == 'M');

% Calculate the correlation matrix including "id" and "diagnosis"
correlationMatrix = corr(dataWithoutId{:, :});

% Customize the heatmap
figure;

% Modify variable names for subscript-like appearance
variableNames = dataWithoutId.Properties.VariableNames;
for i = 1:length(variableNames)
    variableNames{i} = strrep(variableNames{i}, '_', '\_'); % Replace underscores with escaped underscores
end

% Create the heatmap using imagesc
h = imagesc(correlationMatrix);
colormap(jet);
colorbar;

% Set axis labels with modified variable names
xticks(1:length(variableNames));
xticklabels(variableNames);
yticks(1:length(variableNames));
yticklabels(variableNames);

% Rotate axis labels for better visibility
xtickangle(90);

% Set axis label font size
set(gca, 'FontSize', 8);

% Add text labels for each cell in the heatmap
for i = 1:size(correlationMatrix, 1)
    for j = 1:size(correlationMatrix, 2)
        text(j, i, sprintf('%.2f', correlationMatrix(i, j)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 8);
    end
end

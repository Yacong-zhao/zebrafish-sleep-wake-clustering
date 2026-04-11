%% Zebrafish physiological data clustering analysis
clear; close all; clc;

% ===================== 全局变量定义（修复变量未识别问题）=====================
% 1. 聚类颜色（全脚本可访问）
gray_color = [0.5, 0.5, 0.5];    % 灰色聚类 RGB值
purple_color = [0.5, 0, 0.5];    % 紫色聚类 RGB值

% 2. 初始化关键参数（可选，提升代码健壮性）
window_size = 5;                 % 默认窗口大小
n_clusters = 2;                  % 默认聚类数
max_iters = 100;                 % K-means最大迭代次数
rng(42);                         % 固定随机种子，保证结果可复现
% ===========================================================================

%% 1. Data loading
disp('=== Zebrafish Physiological Data Clustering Analysis ===');

% Set default file path
default_path = 'C:\Users\Administrator\Desktop\sleep3.csv';
file_path = input('Enter CSV file path (press Enter for default): ', 's');
if isempty(file_path)
    file_path = default_path;
    fprintf('Using default path: %s\n', file_path);
end

% Read CSV or generate sample data
try
    data = readtable(file_path, 'VariableNamingRule', 'preserve');
    fprintf('Data loaded: %dx%d\n', size(data));
    disp(head(data, 3));
catch
    fprintf('File loading failed, generating sample data...\n');
    n_samples = 1000;
    time = linspace(0, 100, n_samples)';
    heart_rate = 80 + 20 * sin(time/10) + randn(n_samples, 1) * 5;
    emg = 50 + 30 * sin(time/8 + 2) + randn(n_samples, 1) * 8;
    calcium = 100 + 40 * sin(time/15 + 5) + randn(n_samples, 1) * 10;
    data = table(time, heart_rate, emg, calcium, ...
                 'VariableNames', {'Time', 'HeartRate_bpm', 'EMG_uV', 'Calcium_au'});
end

%% 检查钙活动数据特性（仅展示信息，不修改数据）
fprintf('\n=== 钙活动数据检查 ===\n');
if ismember('Calcium_au', data.Properties.VariableNames) || ...
   any(contains(data.Properties.VariableNames, {'钙', 'Calcium', 'calcium'}, 'IgnoreCase', true))
    
    % 找到钙活动列
    calcium_cols = contains(data.Properties.VariableNames, {'钙', 'Calcium', 'calcium'}, 'IgnoreCase', true);
    calcium_col_name = data.Properties.VariableNames{find(calcium_cols, 1)};
    calcium_data = table2array(data(:, calcium_col_name));
    
    fprintf('钙活动特征: %s\n', calcium_col_name);
    fprintf('  数据点数: %d\n', length(calcium_data));
    fprintf('  最小值: %.2f\n', min(calcium_data));
    fprintf('  最大值: %.2f\n', max(calcium_data));
    fprintf('  均值: %.2f\n', mean(calcium_data));
    fprintf('  中位数: %.2f\n', median(calcium_data));
    fprintf('  标准差: %.2f\n', std(calcium_data));
    fprintf('  偏度: %.2f\n', skewness(calcium_data));
    fprintf('  峰度: %.2f\n', kurtosis(calcium_data));
    fprintf('  NaN值数量: %d\n', sum(isnan(calcium_data)));
    fprintf('  Inf值数量: %d\n', sum(isinf(calcium_data)));
    fprintf('  零值数量: %d\n', sum(calcium_data == 0));
    
    % 仅展示数据范围，不建议转换
    if min(calcium_data) >= 0
        fprintf('  数据范围: 非负（将直接使用原始数值，不做转换）\n');
    else
        fprintf('  数据范围: 包含负值（将直接使用原始数值，不做转换）\n');
    end
end

%% 2. Time window feature extraction
disp('=== Time Window Feature Extraction ===');

window_size = input('Enter time window size in seconds (default: 5): ');
if isempty(window_size) || window_size <= 0
    window_size = 5;
end
fprintf('Using window size: %d seconds\n', window_size);

time_data = table2array(data(:, 1));
feature_columns = data.Properties.VariableNames(2:end);
features_raw = table2array(data(:, 2:end));
n_features_raw = size(features_raw, 2);

total_time = max(time_data) - min(time_data);
n_windows = floor(total_time / window_size);

if n_windows < 2
    error('Time window size is too large. Need at least 2 windows.');
end

window_edges = linspace(min(time_data), max(time_data), n_windows + 1);
window_centers = (window_edges(1:end-1) + window_edges(2:end)) / 2;

window_indices = discretize(time_data, window_edges);
window_indices(isnan(window_indices)) = n_windows;

window_features = zeros(n_windows, n_features_raw);
window_counts = zeros(n_windows, 1);

for w = 1:n_windows
    in_window = (window_indices == w);
    window_counts(w) = sum(in_window);
    
    if window_counts(w) > 0
        window_features(w, :) = max(features_raw(in_window, :), [], 1);
    else
        if w > 1
            window_features(w, :) = window_features(w-1, :);
        else
            window_features(w, :) = mean(features_raw, 1);
        end
    end
end

% Update variables
feature_names = feature_columns;
features = window_features;
n_features = n_features_raw;
n_samples = n_windows;
time_data = window_centers;

%% 3. Data preprocessing - 简化版，仅保留异常值处理，不转换钙活动数据
disp('=== Data Preprocessing (简化版) ===');

% Remove rows with missing values
valid_rows = ~any(isnan(features), 2);
features = features(valid_rows, :);
time_data = time_data(valid_rows);
n_samples = size(features, 1);

% 仅处理钙活动数据的异常值，不进行任何数值转换
calcium_processed = false;
for i = 1:n_features
    feature_name = feature_names{i};
    
    % 检查是否是钙活动数据
    if contains(feature_name, {'钙', 'Calcium', 'calcium'}, 'IgnoreCase', true)
        fprintf('\n处理钙活动特征: %s（仅异常值替换，不转换数值）\n', feature_name);
        
        calcium_data = features(:, i);  % 直接使用原始钙活动数值
        
        % 1. 检查并替换异常值（仅必要处理，不修改原始数值分布）
        Q1 = prctile(calcium_data, 25);
        Q3 = prctile(calcium_data, 75);
        IQR = Q3 - Q1;
        lower_bound = Q1 - 3 * IQR;
        upper_bound = Q3 + 3 * IQR;
        
        outliers = calcium_data < lower_bound | calcium_data > upper_bound;
        fprintf('  异常值检测: %d个点 (%.1f%%) 超出 [%.2f, %.2f]\n', ...
                sum(outliers), sum(outliers)/length(calcium_data)*100, lower_bound, upper_bound);
        
        % 用中位数替换异常值（仅修复异常值，不改变其他原始数值）
        if sum(outliers) > 0
            calcium_data(outliers) = median(calcium_data(~outliers));
            features(:, i) = calcium_data;
            fprintf('  已替换异常值，其余数值保持原始值不变\n');
        else
            fprintf('  无异常值，直接使用原始数值\n');
        end
        
        % 2. 仅展示偏度信息，不进行任何转换
        skewness_val = skewness(calcium_data);
        fprintf('  偏度: %.2f（不进行数据转换）\n', skewness_val);
        
        % 3. 标准化仅用于聚类计算，原始数值仍保留用于阈值输出
        col_mean = mean(features(:, i));
        col_std = std(features(:, i));
        fprintf('  标准化参数（仅聚类使用）: 均值=%.2f, 标准差=%.2f\n', col_mean, col_std);
        
        calcium_processed = true;
    end
end

% Standardization（仅用于聚类计算，不改变原始数值的阈值输出）
features_std = zeros(size(features));
for i = 1:n_features
    col_mean = mean(features(:, i));
    col_std = std(features(:, i));
    if col_std > 0
        features_std(:, i) = (features(:, i) - col_mean) / col_std;
    else
        features_std(:, i) = features(:, i) - col_mean;
    end
end

if calcium_processed
    fprintf('\n钙活动数据预处理完成：仅异常值替换，全程使用原始数值\n');
end

%% 4. Manual K-means clustering (2 classes)
disp('=== Manual K-means Clustering ===');
n_clusters = 2;

if n_samples <= n_clusters
    error('Not enough data points for clustering.');
end

rng(42); % Set random seed
centroid_indices = randperm(n_samples, n_clusters);
centroids = features_std(centroid_indices, :);

max_iters = 100;
labels = zeros(n_samples, 1);

for iter = 1:max_iters
    % Assign points to centroids
    for i = 1:n_samples
        distances = zeros(1, n_clusters);
        for k = 1:n_clusters
            diff = features_std(i, :) - centroids(k, :);
            distances(k) = sqrt(sum(diff.^2));
        end
        [~, labels(i)] = min(distances);
    end
    
    % Update centroids
    new_centroids = zeros(n_clusters, n_features);
    for k = 1:n_clusters
        cluster_points = features_std(labels == k, :);
        if size(cluster_points, 1) > 0
            new_centroids(k, :) = mean(cluster_points, 1);
        else
            new_centroids(k, :) = features_std(randi(n_samples), :);
        end
    end
    
    % Check convergence
    centroid_change = max(abs(new_centroids(:) - centroids(:)));
    if centroid_change < 1e-6
        fprintf('Converged after %d iterations\n', iter);
        break;
    end
    centroids = new_centroids;
end

% 补全缺失的聚类索引和大小变量（关键修复）
gray_idx = labels == 1;
purple_idx = labels == 2;
gray_size = sum(gray_idx);
purple_size = sum(purple_idx);
gray_percentage = gray_size / n_samples * 100;
purple_percentage = purple_size / n_samples * 100;

%% 5. PCA Analysis
disp('=== PCA Analysis ===');

% Center the data
X_centered = features_std - mean(features_std, 1);

% Calculate covariance matrix
cov_matrix = (X_centered' * X_centered) / (n_samples - 1);

% Eigen decomposition
[eigenvectors, eigenvalues] = eig(cov_matrix);
eigenvalues = diag(eigenvalues);

% Sort eigenvalues and eigenvectors
[eigenvalues, idx] = sort(eigenvalues, 'descend');
eigenvectors = eigenvectors(:, idx);

% Project data onto first 2 principal components
pc_scores = X_centered * eigenvectors(:, 1:2);

% Calculate variance explained
total_variance = sum(eigenvalues);
variance_explained = eigenvalues(1:2) / total_variance * 100;

% Calculate cluster statistics in PCA space
centroids_pca = (centroids - mean(features_std, 1)) * eigenvectors(:, 1:2);

%% 6.1 精简版：计算三个特征的聚类阈值（仅使用原始数值）
disp('=== Calculating Cluster Thresholds for Three Key Metrics ===');

% 1. 选择前三个核心特征
n_threshold_features = min(3, n_features);
threshold_feature_idx = 1:n_threshold_features;
threshold_feature_names = feature_names(threshold_feature_idx);

% 2. 分离两聚类的原始特征数据（完全使用原始数值，无转换）
gray_features_original = features(labels == 1, threshold_feature_idx);  % Gray聚类原始数据
purple_features_original = features(labels == 2, threshold_feature_idx);% Purple聚类原始数据

% 3. 逐特征计算均值和阈值（仅基于原始数值，无额外计算）
gray_means = zeros(n_threshold_features, 1);    % Gray聚类均值（列向量）
purple_means = zeros(n_threshold_features, 1);  % Purple聚类均值（列向量）
thresholds_original = zeros(n_threshold_features, 1);  % 原始尺度阈值（列向量）

for f = 1:n_threshold_features
    % 直接使用原始数值计算阈值（仅两聚类均值中点，无其他计算）
    gray_means(f) = mean(gray_features_original(:, f));
    purple_means(f) = mean(purple_features_original(:, f));
    thresholds_original(f) = (gray_means(f) + purple_means(f)) / 2;
    
    % 输出简洁结果（强调使用原始数值）
    fprintf('[%s] 阈值标记（原始数值）：Gray均值=%.2f, Purple均值=%.2f, 阈值=%.2f\n', ...
            threshold_feature_names{f}, gray_means(f), purple_means(f), thresholds_original(f));
end

% 4. 阈值可视化（仅标记阈值位置，使用原始数值）
disp('=== Visualizing Cluster Thresholds ===');
figure('Position', [100, 100, 1200, 800], 'Name', 'Cluster Thresholds (Three Features)', 'Color', 'white');

for f = 1:n_threshold_features
    subplot(1, n_threshold_features, f);
    hold on; grid on; box on;
    
    % 绘制两聚类的原始数据分布（无转换）
    all_data = [gray_features_original(:, f); purple_features_original(:, f)];
    bin_edges = linspace(min(all_data), max(all_data), 25);
    
    % Gray聚类直方图
    histogram(gray_features_original(:, f), bin_edges, ...
              'FaceColor', gray_color, 'FaceAlpha', 0.6, 'EdgeColor', 'none', ...
              'Normalization', 'probability', 'DisplayName', 'Gray Cluster');
    % Purple聚类直方图
    histogram(purple_features_original(:, f), bin_edges, ...
              'FaceColor', purple_color, 'FaceAlpha', 0.6, 'EdgeColor', 'none', ...
              'Normalization', 'probability', 'DisplayName', 'Purple Cluster');
    
    % 标记阈值（红色竖线，基于原始数值）
    threshold_val = thresholds_original(f);
    plot([threshold_val, threshold_val], ylim, 'r-', 'LineWidth', 3, ...
         'DisplayName', sprintf('Threshold = %.2f', threshold_val));
    
    % 图表标注
    xlabel(threshold_feature_names{f}, 'FontSize', 12, 'Interpreter', 'none');
    ylabel('Probability', 'FontSize', 12);
    title(sprintf('%s Cluster Threshold (Original Values)', threshold_feature_names{f}), 'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 10);
end

sgtitle('Three Key Features - Cluster Threshold (Original Values)', 'FontSize', 16, 'FontWeight', 'bold');
saveas(gcf, 'three_features_cluster_thresholds.png');
fprintf('✓ Threshold plot saved: three_features_cluster_thresholds.png\n');

% 5. 导出简洁的阈值表格（仅原始数值）
disp('=== Exporting Threshold Table ===');
% 方法1：逐行创建表格（最稳妥，避免维度问题）
threshold_table = table( ...
    string(threshold_feature_names), ...  % 特征名（列向量）
    gray_means, ...                       % Gray均值（原始数值）
    purple_means, ...                     % Purple均值（原始数值）
    thresholds_original, ...              % 阈值（原始数值）
    repmat("Mean Midpoint (Original Values)", n_threshold_features, 1), ...  % 方法（列向量）
    'VariableNames', { ...
        'Feature_Name', ...
        'Gray_Cluster_Mean', ...
        'Purple_Cluster_Mean', ...
        'Cluster_Threshold', ...
        'Threshold_Method' ...
    });

% 保存表格（仅核心信息，原始数值）
writetable(threshold_table, 'three_features_thresholds.csv');
fprintf('✓ Threshold table saved: three_features_thresholds.csv\n');

% 显示最终阈值结果（简洁版）
fprintf('\n=== Final Threshold Summary (Original Values) ===\n');
disp(threshold_table);

%% 7. Create visualizations（保留基础可视化，使用原始数值）
disp('=== Creating Visualizations ===');

% Figure 1: Combined cluster distribution
figure('Position', [100, 100, 800, 700], 'Name', 'Combined Cluster Distribution', ...
       'Color', 'white');

hold on;

% Gray cluster
if gray_size >= 3
    gray_data = pc_scores(gray_idx, :);
    [ellipse_x_gray, ellipse_y_gray] = calculate_95percent_ellipse(gray_data, 100);
    if ~isempty(ellipse_x_gray)
        plot(ellipse_x_gray, ellipse_y_gray, 'Color', gray_color * 0.6, ...
             'LineWidth', 2.5, 'LineStyle', '--', 'DisplayName', 'Gray 95% CI');
    end
    scatter(gray_data(:, 1), gray_data(:, 2), 60, gray_color, 'filled', ...
            'MarkerFaceAlpha', 0.6, 'DisplayName', sprintf('Gray (n=%d)', gray_size));
end

% Purple cluster
if purple_size >= 3
    purple_data = pc_scores(purple_idx, :);
    [ellipse_x_purple, ellipse_y_purple] = calculate_95percent_ellipse(purple_data, 100);
    if ~isempty(ellipse_x_purple)
        plot(ellipse_x_purple, ellipse_y_purple, 'Color', purple_color * 0.6, ...
             'LineWidth', 2.5, 'LineStyle', '--', 'DisplayName', 'Purple 95% CI');
    end
    scatter(purple_data(:, 1), purple_data(:, 2), 60, purple_color, 'filled', ...
            'MarkerFaceAlpha', 0.6, 'DisplayName', sprintf('Purple (n=%d)', purple_size));
end

% Centroids
gray_center = centroids_pca(1, :);
purple_center = centroids_pca(2, :);
plot(gray_center(1), gray_center(2), 's', 'MarkerSize', 12, ...
     'MarkerFaceColor', gray_color, 'MarkerEdgeColor', 'black', ...
     'LineWidth', 2, 'DisplayName', 'Gray Centroid');
plot(purple_center(1), purple_center(2), 's', 'MarkerSize', 12, ...
     'MarkerFaceColor', purple_color, 'MarkerEdgeColor', 'black', ...
     'LineWidth', 2, 'DisplayName', 'Purple Centroid');

% Labels and title
xlabel(sprintf('PC1 (%.1f%%)', variance_explained(1)), 'FontSize', 14, 'FontWeight', 'bold');
ylabel(sprintf('PC2 (%.1f%%)', variance_explained(2)), 'FontSize', 14, 'FontWeight', 'bold');
title('Combined Cluster Distribution with 95% CI', 'FontSize', 16, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 11);
box on;
grid on;

saveas(gcf, 'combined_cluster_distribution_95CI.png');

% Figure 2: Feature distributions by cluster（使用原始数值）
gray_original_features = features(gray_idx, :);
purple_original_features = features(purple_idx, :);
n_plots = min(3, n_features);

figure('Position', [100, 100, 1200, 800], 'Name', 'Feature Distributions by Cluster (Original Values)', ...
       'Color', 'white');

for f = 1:n_plots
    subplot(3, n_plots, f);
    hold on;
    
    all_data = [gray_original_features(:, f); purple_original_features(:, f)];
    bin_edges = linspace(min(all_data), max(all_data), 21);
    
    if gray_size > 0
        histogram(gray_original_features(:, f), bin_edges, ...
                  'FaceColor', gray_color, 'FaceAlpha', 0.6, ...
                  'EdgeColor', gray_color * 0.7, 'LineWidth', 0.5, ...
                  'Normalization', 'probability', 'DisplayName', 'Gray');
    end
    
    if purple_size > 0
        histogram(purple_original_features(:, f), bin_edges, ...
                  'FaceColor', purple_color, 'FaceAlpha', 0.6, ...
                  'EdgeColor', purple_color * 0.7, 'LineWidth', 0.5, ...
                  'Normalization', 'probability', 'DisplayName', 'Purple');
    end
    
    xlabel(feature_names{f}, 'FontSize', 11, 'Interpreter', 'none');
    ylabel('Probability', 'FontSize', 11);
    title(sprintf('%s Distribution (Original Values)', feature_names{f}), 'FontSize', 12);
    legend('Location', 'best', 'FontSize', 10);
    grid on;
end

sgtitle('Feature Distributions by Cluster (Original Values)', 'FontSize', 14);
saveas(gcf, 'feature_distributions_by_cluster.png');

%% 8. 精简版结果导出（仅导出原始数值）
disp('=== Exporting Results ===');

% 8.1 Export feature statistics（基于原始数值）
feature_stats_gray = zeros(n_plots, 4);
feature_stats_purple = zeros(n_plots, 4);

for f = 1:n_plots
    if gray_size > 0
        feature_stats_gray(f, 1) = mean(gray_original_features(:, f));  % 原始数值均值
        feature_stats_gray(f, 2) = std(gray_original_features(:, f));   % 原始数值标准差
        feature_stats_gray(f, 3) = median(gray_original_features(:, f));% 原始数值中位数
        feature_stats_gray(f, 4) = iqr(gray_original_features(:, f));   % 原始数值四分位距
    end
    
    if purple_size > 0
        feature_stats_purple(f, 1) = mean(purple_original_features(:, f));
        feature_stats_purple(f, 2) = std(purple_original_features(:, f));
        feature_stats_purple(f, 3) = median(purple_original_features(:, f));
        feature_stats_purple(f, 4) = iqr(purple_original_features(:, f));
    end
end

% Create statistics table
all_stats_table = table('Size', [n_plots*2, 6], ...
                       'VariableTypes', {'string', 'string', 'double', 'double', 'double', 'double'}, ...
                       'VariableNames', {'Cluster', 'Feature', 'Mean', 'Std', 'Median', 'IQR'});

row = 1;
for f = 1:n_plots
    all_stats_table.Cluster(row) = "Gray";
    all_stats_table.Feature(row) = string(feature_names{f});
    all_stats_table.Mean(row) = feature_stats_gray(f, 1);
    all_stats_table.Std(row) = feature_stats_gray(f, 2);
    all_stats_table.Median(row) = feature_stats_gray(f, 3);
    all_stats_table.IQR(row) = feature_stats_gray(f, 4);
    row = row + 1;
    
    all_stats_table.Cluster(row) = "Purple";
    all_stats_table.Feature(row) = string(feature_names{f});
    all_stats_table.Mean(row) = feature_stats_purple(f, 1);
    all_stats_table.Std(row) = feature_stats_purple(f, 2);
    all_stats_table.Median(row) = feature_stats_purple(f, 3);
    all_stats_table.IQR(row) = feature_stats_purple(f, 4);
    row = row + 1;
end

writetable(all_stats_table, 'feature_statistics_by_cluster.csv');
fprintf('Feature statistics (original values) saved to feature_statistics_by_cluster.csv\n');

% 8.2 导出原始数据+聚类标签（完全保留原始数值）
disp('=== Exporting Original Data with Cluster Labels ===');

% 获取原始时间序列数据
original_time = table2array(data(:, 1));
original_features = table2array(data(:, 2:end));
n_original_samples = length(original_time);

fprintf('Original data: %d time points, %d features (all original values)\n', n_original_samples, n_features_raw);

% 为每个原始时间点分配聚类标签
fprintf('Assigning cluster labels to original time points...\n');

% 重新计算每个原始时间点所属的窗口
original_window_indices = discretize(original_time, window_edges);
original_window_indices(isnan(original_window_indices)) = n_windows;

% 为每个原始时间点分配聚类标签
original_labels = zeros(n_original_samples, 1);
for i = 1:n_original_samples
    window_idx = original_window_indices(i);
    if window_idx >= 1 && window_idx <= length(labels)
        original_labels(i) = labels(window_idx);
    else
        original_labels(i) = 1; % 默认值
    end
end

% 创建颜色标签
original_cluster_colors = cell(n_original_samples, 1);
for i = 1:n_original_samples
    if original_labels(i) == 1
        original_cluster_colors{i} = 'Gray';
    else
        original_cluster_colors{i} = 'Purple';
    end
end

% 创建原始数据+聚类标签的表格（完全保留原始数值）
original_data_table = table();
original_data_table.OriginalTime = original_time;

% 添加原始特征值（无任何转换）
for f = 1:n_features_raw
    feature_name = feature_columns{f};
    feature_name = matlab.lang.makeValidName(feature_name);
    original_data_table.(feature_name) = original_features(:, f);
end

% 添加聚类标签
original_data_table.ClusterLabel = original_labels;
original_data_table.ClusterColor = original_cluster_colors;

% 添加窗口信息
original_data_table.WindowIndex = original_window_indices;
window_center_times = zeros(n_original_samples, 1);
for i = 1:n_original_samples
    window_idx = original_window_indices(i);
    if window_idx >= 1 && window_idx <= length(window_centers)
        window_center_times(i) = window_centers(window_idx);
    else
        window_center_times(i) = NaN;
    end
end
original_data_table.WindowCenterTime = window_center_times;

% 导出到CSV（添加错误处理）
max_retries = 3;
retry_count = 0;
file_saved = false;

while retry_count < max_retries && ~file_saved
    try
        writetable(original_data_table, 'original_data_with_clusters.csv');
        fprintf('✓ Original data (all values) with cluster labels saved to original_data_with_clusters.csv\n');
        file_saved = true;
    catch ME
        retry_count = retry_count + 1;
        fprintf('✗ Attempt %d failed: %s\n', retry_count, ME.message);
        if retry_count < max_retries
            fclose('all');
            pause(2);
        end
    end
end

%% 9. 最终输出
fprintf('\n');
fprintf('============================================================\n');
fprintf('✅ ANALYSIS COMPLETED SUCCESSFULLY! (Original Values Only)\n');
fprintf('============================================================\n\n');
fprintf('Key Results Summary:\n');
fprintf('   Windows analyzed: %d\n', n_samples);
fprintf('   Gray cluster: %.1f%%\n', gray_percentage);
fprintf('   Purple cluster: %.1f%%\n', purple_percentage);
fprintf('   PCA variance explained: %.1f%%\n\n', sum(variance_explained));

fprintf('核心阈值输出文件（原始数值）:\n');
fprintf('   1. three_features_cluster_thresholds.png - 三个特征阈值标记图\n');
fprintf('   2. three_features_thresholds.csv - 三个特征阈值表格\n\n');

fprintf('其他输出文件（原始数值）:\n');
fprintf('   1. original_data_with_clusters.csv - 原始数据+聚类标签\n');
fprintf('   2. feature_statistics_by_cluster.csv - 特征统计信息\n');
fprintf('   3. combined_cluster_distribution_95CI.png - PCA聚类分布图\n\n');

%% ========================================================================
% 辅助函数定义（保留核心辅助函数）
% ========================================================================

function [ellipse_x, ellipse_y] = calculate_95percent_ellipse(data, n_points)
    % 计算95%置信椭圆
    if size(data, 1) < 2
        ellipse_x = [];
        ellipse_y = [];
        return;
    end
    
    mu = mean(data, 1);
    sigma = cov(data);
    
    % 检查协方差矩阵
    if any(isnan(sigma(:))) || any(isinf(sigma(:)))
        ellipse_x = [];
        ellipse_y = [];
        return;
    end
    
    [V, D] = eig(sigma);
    chi2_val = sqrt(5.991); % sqrt(chi2inv(0.95, 2))
    
    theta = linspace(0, 2*pi, n_points);
    circle = [cos(theta); sin(theta)];
    
    ellipse = V * (chi2_val * sqrt(D)) * circle;
    
    % 确保正确的维度
    if size(ellipse, 1) >= 2
        ellipse_x = ellipse(1, :) + mu(1);
        ellipse_y = ellipse(2, :) + mu(2);
    else
        ellipse_x = [];
        ellipse_y = [];
    end
end
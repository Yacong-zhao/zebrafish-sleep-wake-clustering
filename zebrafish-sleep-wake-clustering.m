%% Zebrafish Sleep-Wake Classifier Training (KNN分类器)
% 根据CSV文件列顺序: Time, HeartRate_norm, EMG_norm, Calcium_norm, Label, HeartRate_raw, EMG_raw, Calcium_raw
clear; close all; clc;

%% 1. 加载数据
disp('=== Step 1: Load Data ===');
file_path = 'C:\Users\Administrator\Desktop\sleep3.csv';
data = readtable(file_path);

fprintf('✓ 数据加载成功: %d 行, %d 列\n', size(data,1), size(data,2));

% 获取列名
col_names = data.Properties.VariableNames;
fprintf('\n数据列名:\n');
for i = 1:length(col_names)
    fprintf('  %d. %s\n', i, col_names{i});
end

%% 2. 根据列顺序提取数据
disp('=== Step 2: Extract Data ===');

% 根据你的CSV列顺序定义索引
% 1:Time, 2:HeartRate_norm, 3:EMG_norm, 4:Calcium_norm, 5:Label, 6:HeartRate_raw, 7:EMG_raw, 8:Calcium_raw

time_col = 1;                    % 时间列
norm_cols_idx = [2, 3, 4];       % 归一化数据列（3列）
label_col = 5;                   % Label列
raw_cols_idx = [6, 7, 8];        % 原始数据列（3列）

% 提取数据
time_data = data{:, time_col};
norm_data = data{:, norm_cols_idx};    % 3列归一化数据
raw_data = data{:, raw_cols_idx};      % 3列原始数据

% 获取列名
norm_cols = col_names(norm_cols_idx);
raw_cols = col_names(raw_cols_idx);

fprintf('✓ 时间列: %s\n', col_names{time_col});
fprintf('✓ 归一化数据列: %s\n', strjoin(norm_cols, ', '));
fprintf('  维度: %d 行 x %d 列\n', size(norm_data,1), size(norm_data,2));
fprintf('✓ 原始数据列: %s\n', strjoin(raw_cols, ', '));
fprintf('  维度: %d 行 x %d 列\n', size(raw_data,1), size(raw_data,2));

% 提取Label
raw_labels = data{:, label_col};

% 转换标签
label_values = zeros(size(raw_labels, 1), 1);
for i = 1:size(raw_labels, 1)
    if iscell(raw_labels)
        current_label = raw_labels{i};
        if ischar(current_label)
            lower_label = lower(current_label);
            if strcmp(lower_label, 'sleep')
                label_values(i) = 1;
            elseif strcmp(lower_label, 'wake')
                label_values(i) = 2;
            else
                label_values(i) = 0;
            end
        elseif isnumeric(current_label)
            if current_label == 1
                label_values(i) = 1;
            elseif current_label == 2
                label_values(i) = 2;
            else
                label_values(i) = 0;
            end
        end
    else
        current_label = raw_labels(i);
        if current_label == 1
            label_values(i) = 1;
        elseif current_label == 2
            label_values(i) = 2;
        else
            label_values(i) = 0;
        end
    end
end

fprintf('✓ Label列: %s\n', col_names{label_col});
fprintf('  标签分布:\n');
fprintf('    Sleep (1):  %d 个点\n', sum(label_values == 1));
fprintf('    Wake (2):   %d 个点\n', sum(label_values == 2));
fprintf('    Unlabeled (0): %d 个点\n', sum(label_values == 0));

n_features = size(raw_data, 2);  % 特征数量 = 3

%% 3. 窗口化参数
disp('=== Step 3: Windowing Setup ===');
window_size = input('Enter window size (seconds, default: 5): ');
if isempty(window_size), window_size = 5; end
fprintf('✓ 窗口大小: %d 秒\n', window_size);

%% 4. 窗口化特征提取
disp('=== Step 4: Windowing ===');
total_time = max(time_data) - min(time_data);
n_windows = floor(total_time / window_size);

fprintf('总时间: %.1f 秒, 窗口数: %d\n', total_time, n_windows);

if n_windows < 2
    error('数据时长太短，请减小窗口大小');
end

window_edges = linspace(min(time_data), max(time_data), n_windows + 1);
window_centers = (window_edges(1:end-1) + window_edges(2:end)) / 2;
window_indices = discretize(time_data, window_edges);
window_indices(isnan(window_indices)) = n_windows;

% 初始化窗口数据
window_norm = zeros(n_windows, n_features);
window_raw = zeros(n_windows, n_features);
window_labels = zeros(n_windows, 1);

for w = 1:n_windows
    in_window = (window_indices == w);
    if sum(in_window) > 0
        % 每个窗口取最大值
        window_norm(w, :) = max(norm_data(in_window, :), [], 1);
        window_raw(w, :) = max(raw_data(in_window, :), [], 1);
        
        % 窗口的标签（多数决）
        sleep_count = sum(label_values(in_window) == 1);
        wake_count = sum(label_values(in_window) == 2);
        
        if sleep_count > wake_count && sleep_count > 0
            window_labels(w) = 1;  % Sleep
        elseif wake_count > sleep_count && wake_count > 0
            window_labels(w) = 2;  % Wake
        else
            window_labels(w) = 0;  % Unlabeled
        end
    elseif w > 1
        window_norm(w, :) = window_norm(w-1, :);
        window_raw(w, :) = window_raw(w-1, :);
        window_labels(w) = window_labels(w-1);
    else
        window_norm(w, :) = mean(norm_data, 1);
        window_raw(w, :) = mean(raw_data, 1);
        window_labels(w) = 0;
    end
end

fprintf('✓ 生成 %d 个窗口\n', n_windows);
fprintf('  window_raw 维度: %d x %d\n', size(window_raw,1), size(window_raw,2));
fprintf('  window_norm 维度: %d x %d\n', size(window_norm,1), size(window_norm,2));

fprintf('\n窗口锚点标签分布:\n');
fprintf('  Sleep (1):  %d 个窗口\n', sum(window_labels == 1));
fprintf('  Wake (2):   %d 个窗口\n', sum(window_labels == 2));
fprintf('  Unlabeled (0): %d 个窗口\n', sum(window_labels == 0));

%% 5. K-means聚类
disp('=== Step 5: K-means Clustering ===');
n_clusters = 2;
rng(42);

[labels, centroids] = kmeans(window_norm, n_clusters, ...
    'Distance', 'sqeuclidean', ...
    'Replicates', 10, ...
    'MaxIter', 100);

%% 6. 锚点定义（使用CSV中的Label列）
disp('=== Step 6: Anchor-based Label Assignment ===');

if sum(window_labels ~= 0) > 0
    fprintf('\n使用CSV文件中的Label列作为锚点...\n');
    
    labeled_windows = find(window_labels ~= 0);
    anchor_labels_gt = window_labels(labeled_windows);
    cluster_labels_at_anchor = labels(labeled_windows);
    
    fprintf('✓ 找到 %d 个有锚点标签的窗口\n', length(labeled_windows));
    fprintf('  Sleep锚点: %d 个\n', sum(anchor_labels_gt == 1));
    fprintf('  Wake锚点: %d 个\n', sum(anchor_labels_gt == 2));
    
    % 计算两种对应方式的匹配度
    % 方案A：聚类1=Sleep, 聚类2=Wake
    match_count_A = sum(cluster_labels_at_anchor == 1 & anchor_labels_gt == 1) + ...
                    sum(cluster_labels_at_anchor == 2 & anchor_labels_gt == 2);
    
    % 方案B：聚类1=Wake, 聚类2=Sleep
    match_count_B = sum(cluster_labels_at_anchor == 1 & anchor_labels_gt == 2) + ...
                    sum(cluster_labels_at_anchor == 2 & anchor_labels_gt == 1);
    
    fprintf('\n锚点匹配分析:\n');
    fprintf('  方案A (聚类1=Sleep, 聚类2=Wake): %d/%d 匹配 (%.1f%%)\n', ...
        match_count_A, length(labeled_windows), match_count_A/length(labeled_windows)*100);
    fprintf('  方案B (聚类1=Wake, 聚类2=Sleep): %d/%d 匹配 (%.1f%%)\n', ...
        match_count_B, length(labeled_windows), match_count_B/length(labeled_windows)*100);
    
    % 选择匹配度更高的方案
    if match_count_A >= match_count_B
        fprintf('\n✓ 选择方案A: 聚类1 = Sleep, 聚类2 = Wake\n');
    else
        fprintf('\n✓ 选择方案B: 聚类1 = Wake, 聚类2 = Sleep\n');
        labels = 3 - labels;
        centroids = flipud(centroids);
    end
    
    anchor_accuracy = max(match_count_A, match_count_B) / length(labeled_windows) * 100;
    fprintf('✓ 锚点匹配准确率: %.1f%%\n', anchor_accuracy);
    
else
    fprintf('\n⚠️ 未找到有效的锚点标签，使用自动阈值判断...\n');
    if mean(window_raw(labels==1, 1)) < mean(window_raw(labels==2, 1))
        labels = 3 - labels;
        centroids = flipud(centroids);
        fprintf('✓ 已根据特征均值交换聚类\n');
    end
end

% 最终标签
sleep_idx = (labels == 1);
wake_idx = (labels == 2);

fprintf('\n=== 最终聚类结果 ===\n');
fprintf('Sleep (睡眠): %d 窗口 (%.1f%%)\n', sum(sleep_idx), sum(sleep_idx)/n_windows*100);
fprintf('Wake (清醒): %d 窗口 (%.1f%%)\n', sum(wake_idx), sum(wake_idx)/n_windows*100);

% 显示各聚类特征均值
fprintf('\n各聚类特征均值（原始数据）:\n');
fprintf('%-20s %12s %12s\n', 'Feature', 'Sleep Mean', 'Wake Mean');
for i = 1:n_features
    fprintf('%-20s %12.2f %12.2f\n', raw_cols{i}, ...
        mean(window_raw(sleep_idx, i)), mean(window_raw(wake_idx, i)));
end

%% 7. 训练KNN分类器
disp('=== Step 7: Train KNN Classifier ===');

X_train = window_raw;
y_train = labels;

fprintf('训练数据: %d 样本, %d 特征\n', size(X_train,1), size(X_train,2));

% 划分训练集和验证集（80%训练，20%验证）
n_samples = size(X_train,1);
n_train = round(0.8 * n_samples);
rand_indices = randperm(n_samples);
X_train_cv = X_train(rand_indices(1:n_train), :);
y_train_cv = y_train(rand_indices(1:n_train));
X_val = X_train(rand_indices(n_train+1:end), :);
y_val = y_train(rand_indices(n_train+1:end));

% 训练KNN分类器（尝试不同的K值）
k_values = [1, 3, 5, 7, 9, 11];
best_k = 5;
best_k_acc = 0;

fprintf('\n尝试不同的K值:\n');
for k = k_values
    knn_model = fitcknn(X_train_cv, y_train_cv, 'NumNeighbors', k);
    y_pred = predict(knn_model, X_val);
    acc = sum(y_pred == y_val) / length(y_val);
    fprintf('  K=%d: 验证准确率 = %.1f%%\n', k, acc*100);
    if acc > best_k_acc
        best_k_acc = acc;
        best_k = k;
    end
end

fprintf('\n✓ 最佳K值: %d (验证准确率: %.1f%%)\n', best_k, best_k_acc*100);

% 使用最佳K值训练最终模型
final_model = fitcknn(X_train, y_train, 'NumNeighbors', best_k);
fprintf('✓ 最终KNN模型已训练 (K=%d)\n', best_k);

%% 8. 5折交叉验证（手动实现）
disp('=== Step 8: 5-Fold Cross Validation ===');

k_folds = 5;
n_samples = size(X_train, 1);

% 手动创建5折索引
indices = zeros(n_samples, 1);
fold_size = floor(n_samples / k_folds);
for fold = 1:k_folds-1
    indices((fold-1)*fold_size + 1:fold*fold_size) = fold;
end
indices((k_folds-1)*fold_size + 1:end) = k_folds;

% 随机打乱
rand_idx = randperm(n_samples);
indices = indices(rand_idx);

fold_accuracies = zeros(k_folds, 1);

for fold = 1:k_folds
    test_idx = (indices == fold);
    train_idx = ~test_idx;
    
    X_train_fold = X_train(train_idx, :);
    y_train_fold = y_train(train_idx);
    X_test_fold = X_train(test_idx, :);
    y_test_fold = y_train(test_idx);
    
    temp_model = fitcknn(X_train_fold, y_train_fold, 'NumNeighbors', best_k);
    y_pred_fold = predict(temp_model, X_test_fold);
    
    fold_accuracies(fold) = sum(y_pred_fold == y_test_fold) / length(y_test_fold);
    fprintf('  Fold %d: %.1f%% (%d/%d)\n', fold, fold_accuracies(fold)*100, ...
        sum(y_pred_fold == y_test_fold), length(y_test_fold));
end

cv_acc = mean(fold_accuracies) * 100;
cv_std = std(fold_accuracies) * 100;
fprintf('\n5折交叉验证结果: %.1f%% ± %.1f%%\n', cv_acc, cv_std);

%% 9. 保存分类器
disp('=== Step 9: Save Classifier ===');

classifier = struct();
classifier.version = '5.0';
classifier.training_date = datestr(now);
classifier.classifier_type = 'KNN';
classifier.k_neighbors = best_k;
classifier.window_size = window_size;
classifier.feature_names = raw_cols;
classifier.n_features = n_features;
classifier.model = final_model;
classifier.performance.validation_accuracy = best_k_acc;
classifier.performance.cv_accuracy = cv_acc;
classifier.performance.cv_std = cv_std;
classifier.performance.fold_accuracies = fold_accuracies;
classifier.anchor_info.sleep_percentage = sum(sleep_idx)/n_windows*100;
classifier.anchor_info.wake_percentage = sum(wake_idx)/n_windows*100;
classifier.anchor_info.anchor_accuracy = anchor_accuracy;
classifier.centroids = centroids;

save('classifier_params.mat', 'classifier');
fprintf('✓ 分类器已保存: classifier_params.mat\n');

%% 10. 最终报告
fprintf('\n========================================\n');
fprintf('✅ TRAINING COMPLETED!\n');
fprintf('========================================\n');
fprintf('分类器: KNN (K=%d)\n', best_k);
fprintf('验证准确率: %.1f%%\n', best_k_acc*100);
fprintf('5折交叉验证: %.1f%% ± %.1f%%\n', cv_acc, cv_std);
fprintf('锚点匹配率: %.1f%%\n', anchor_accuracy);
fprintf('\n聚类分布:\n');
fprintf('  Sleep: %.1f%% (%d 窗口)\n', sum(sleep_idx)/n_windows*100, sum(sleep_idx));
fprintf('  Wake: %.1f%% (%d 窗口)\n', sum(wake_idx)/n_windows*100, sum(wake_idx));

function tnl2k_official_eval(toolkit_root, predictions_root, output_json)
% Evaluate one result directory through the official TNL2K utility functions.

utils_root = fullfile(toolkit_root, 'utils');
sequence_file = fullfile(toolkit_root, 'sequence_evaluation_config', ...
                         'TNL2K_testing_set.txt');
annotation_root = fullfile(toolkit_root, 'annos');
addpath(utils_root);

sequence_handle = fopen(sequence_file, 'r');
if sequence_handle < 0
    error('Cannot open official sequence list: %s', sequence_file);
end
sequence_cells = textscan(sequence_handle, '%s');
fclose(sequence_handle);
sequences = sequence_cells{1};

overlap_thresholds = 0:0.05:1;
precision_thresholds = 0:50;
normalized_thresholds = (0:50) / 100;
success_curves = zeros(length(sequences), length(overlap_thresholds));
precision_curves = zeros(length(sequences), length(precision_thresholds));
normalized_curves = zeros(length(sequences), length(normalized_thresholds));

for sequence_index = 1:length(sequences)
    name = sequences{sequence_index};
    annotation = dlmread(fullfile(annotation_root, 'gt_rect', ...
                                  [name '.txt']));
    absent = dlmread(fullfile(annotation_root, 'absent', [name '.txt']));
    if size(absent, 1) > size(annotation, 1)
        absent = absent(1:size(annotation, 1), :);
    elseif size(absent, 1) < size(annotation, 1)
        absent = [absent; zeros(size(annotation, 1) - size(absent, 1), 1)];
    end
    prediction = dlmread(fullfile(predictions_root, [name '.txt']));
    if size(prediction, 1) ~= size(annotation, 1) || size(prediction, 2) ~= 4
        error('%s prediction shape does not match official annotation', name);
    end

    [overlap, center_error] = calc_seq_err_robust( ...
        prediction, annotation, absent, false);
    [~, normalized_error] = calc_seq_err_robust( ...
        prediction, annotation, absent, true);
    denominator = size(annotation, 1) + eps;
    for threshold_index = 1:length(overlap_thresholds)
        success_curves(sequence_index, threshold_index) = ...
            sum(overlap > overlap_thresholds(threshold_index)) / denominator;
    end
    for threshold_index = 1:length(precision_thresholds)
        precision_curves(sequence_index, threshold_index) = ...
            sum(center_error <= precision_thresholds(threshold_index)) / ...
            denominator;
        normalized_curves(sequence_index, threshold_index) = ...
            sum(normalized_error <= normalized_thresholds(threshold_index)) / ...
            denominator;
    end
end

mean_success = mean(success_curves, 1);
mean_precision = mean(precision_curves, 1);
mean_normalized = mean(normalized_curves, 1);
result.success_auc = mean(mean_success);
result.precision_at_20px = mean_precision(21);
result.normalized_precision_auc = mean(mean_normalized);
result.sequence_count = length(sequences);
result.success_curve = mean_success;
result.precision_curve = mean_precision;
result.normalized_precision_curve = mean_normalized;

temporary_output = [output_json '.tmp'];
output_handle = fopen(temporary_output, 'w');
if output_handle < 0
    error('Cannot create output JSON: %s', temporary_output);
end
fprintf(output_handle, '{\n');
fprintf(output_handle, '  "success_auc": %.17g,\n', result.success_auc);
fprintf(output_handle, '  "precision_at_20px": %.17g,\n', ...
        result.precision_at_20px);
fprintf(output_handle, '  "normalized_precision_auc": %.17g,\n', ...
        result.normalized_precision_auc);
fprintf(output_handle, '  "sequence_count": %d,\n', result.sequence_count);
fprintf(output_handle, '  "success_curve": %s,\n', ...
        json_numeric_array(result.success_curve));
fprintf(output_handle, '  "precision_curve": %s,\n', ...
        json_numeric_array(result.precision_curve));
fprintf(output_handle, '  "normalized_precision_curve": %s\n', ...
        json_numeric_array(result.normalized_precision_curve));
fprintf(output_handle, '}\n');
fclose(output_handle);
movefile(temporary_output, output_json, 'f');
end


function encoded = json_numeric_array(values)
values = values(:).';
if isempty(values)
    encoded = '[]';
    return;
end
parts = arrayfun(@(value) sprintf('%.17g', value), ...
                 values, 'UniformOutput', false);
encoded = ['[' strjoin(parts, ',') ']'];
end

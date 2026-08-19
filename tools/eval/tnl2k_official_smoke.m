addpath('data/TNL2K_standard/toolkit/utils');

prediction = [100, 100, 10, 10; NaN, NaN, 0, 0; 12, 10, 10, 10];
groundtruth = [10, 10, 10, 10; 11, 10, 10, 10; 12, 10, 10, 10];
absent = [0; 0; 0];

[overlap, center_error] = calc_seq_err_robust( ...
    prediction, groundtruth, absent, false);

assert(abs(overlap(1) - 1) < 1e-12);
assert(overlap(2) == 0);
assert(abs(overlap(3) - 1) < 1e-12);
assert(center_error(1) == 0);
disp('official_toolkit_synthetic_check=pass');

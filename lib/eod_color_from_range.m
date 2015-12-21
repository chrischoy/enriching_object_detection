function colors = eod_color_from_range(scores, score_range, color_map)
% given a vector of scores and a range (must cover all real value), return colors in matrix form where each row is the color for corresponding score in the scores vector
if nargin < 3
    n_color = numel(color_range);
    color_map = cool(n_color);
end

n_score = numel(scores);
colors = zeros(n_score,3);
for score_idx = 1:n_score
    [~, color_idx] = histc(scores(score_idx), score_range);
    colors(score_idx, :) = color_map(color_idx, :);
end 

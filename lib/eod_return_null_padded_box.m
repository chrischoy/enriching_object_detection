function [null_padded_box, skip_current_image] = eod_return_null_padded_box(formatted_bounding_box, score_threshold, box_col_size)
% Given a formatted box with box(:,end) is score, return null box if the input box is empty.
if numel(formatted_bounding_box) == 0 
    skip_current_image = true;
    null_padded_box = zeros(1,box_col_size);
    null_padded_box( end ) = -inf;
else
    null_padded_box = formatted_bounding_box;
    skip_current_image = false;
end

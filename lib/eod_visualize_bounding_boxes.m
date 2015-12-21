function eod_visualize_bounding_boxes(bounding_boxes, score, text_template, text_tuples, color_range, color_map)
% draw bounding boxes and text

n_box = size(bounding_boxes,1);

for box_idx = 1:n_box
    bounding_box = bounding_boxes(box_idx,:);
    box_text =  eod_bounding_box_text(text_template, text_tuples(box_idx,:));
    box_color = eod_color_from_range(score(box_idx), color_range, color_map);
    eod_visualize_bounding_box_and_text(bounding_box, box_text, box_color);
end

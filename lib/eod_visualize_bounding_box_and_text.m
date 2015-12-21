function eod_visualize_bounding_box_and_text(box_position, box_text, box_color)

box_position = eod_bbox_xy_to_wh(box_position);
% if detector id available (positive number), print it

rectangle('position', box_position,'edgecolor', [0.5 0.5 0.5],'LineWidth',3);
rectangle('position', box_position,'edgecolor', box_color,    'LineWidth',1);
text(box_position(1)+5, box_position(2)-5, box_text, 'BackgroundColor', box_color,...
                        'EdgeColor',[0.5 0.5 0.5],'VerticalAlignment','bottom');


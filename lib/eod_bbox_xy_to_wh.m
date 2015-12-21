function rect_bbox = eod_bbox_xy_to_wh(xy_bbox)
    rect_bbox = xy_bbox(:,1:4) - [0 0 xy_bbox(:,1:2)];
